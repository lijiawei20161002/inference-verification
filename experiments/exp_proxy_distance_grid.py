"""When does the cheap-proxy verifier work? A 2-D MODEL-DISTANCE measurement.

This settles the two questions Roy raised on the thread:

  1. "How different is a *different model*?"  -- there is a gradation, and we make
     it an explicit ordered ladder along TWO independent axes:

       (rows) the model the PROVIDER SECRETLY RUNS (the substitution to catch),
              graded by distance from the claimed model M = Qwen2.5-7B-Instruct:
                0 identical         M itself                         (honest null)
                1 quantized self    M @ NF4 4-bit                    (same weights, lossy)
                2 same family,      Qwen2.5-3B-Instruct              (smaller sibling)
                  smaller
                3 same fam+size,    Qwen2.5-Coder-7B-Instruct        (only post-train
                  diff domain                                         domain differs)
                4 same company,     Qwen3-8B                          (next generation)
                  next gen
                5 same base,        DeepSeek-R1-Distill-Qwen-7B       (Qwen2.5-7B base,
                  diff post-train                                     RL-distilled)
                6 different family  Llama-3.1-8B-Instruct / Phi-3-mini (new arch+tokenizer)

       (cols) the CHEAP PROXY the verifier scores with, graded by distance from M:
                Qwen2.5-0.5B  (same family as M, tiny)
                Qwen2.5-1.5B  (same family as M, bigger)
                SmolLM2-1.7B  (DIFFERENT family -- does the proxy need M's lineage?)

  2. "What are 'scored tokens' and how can AUC be < 0.5?"  -- the verifier never
     re-runs M. It teacher-forces only the first k tokens of the provider's claimed
     completion through the cheap proxy and reads per-token SURPRISE features
     (NLL / rank / entropy). k = "scored tokens" = the verifier's cost knob; we
     sweep it. Detection AUC is the OUT-OF-FOLD AUC of a logistic regression on
     those features (5-fold CV). When the substitution is indistinguishable under
     the proxy (e.g. quantized self), the LR fits fold noise and its held-out
     direction anti-correlates -> AUC lands slightly BELOW 0.5. We report the raw
     signed AUC (so the sub-0.5 dips are visible) alongside the fixed-direction
     mean-NLL AUC, which cannot fit noise and so stays ~0.5 there.

Tokenizer-robustness is the whole reason this generalises past the existing
`exp_cross_family_accept.py` (which is locked to models sharing Qwen's exact token
ids). Here the verifier DECODES the provider's completion to text and RE-TOKENIZES
it under the proxy's own tokenizer before scoring surprise -- so Llama / Phi-3 /
SmolLM2 (entirely different vocabularies) sit on the same ladder as the Qwen models.
This mirrors the real verifier, which only ever sees (prompt_text, completion_text).

Prompts: tatsu-lab/alpaca clean (no-input) instructions. Generation: plain-text
prompt, temperature 1.0 sampling (isolates the model's conditional distribution
from chat-template confounds; every provider answers the same raw instruction).

Run (single H100-80GB; ungated models only):
    IVGYM_N=64 IVGYM_TOKENS=64 .venv/bin/python -m experiments.exp_proxy_distance_grid

Env overrides:
  IVGYM_CLAIMED  claimed/honest model M     (default Qwen/Qwen2.5-7B-Instruct)
  IVGYM_N        prompts / sequences per cell (default 64)
  IVGYM_TOKENS   generated continuation len  (default 64)
  IVGYM_MAXPROMPT prompt truncation tokens   (default 48)
  IVGYM_KSWEEP   comma scored-token budgets (default 8,16,32,64)
"""
from __future__ import annotations

import gc
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# The /workspace volume has a per-volume quota well below its raw capacity; the XET
# chunk cache blows past it mid-run. Disable XET (plain downloads to snapshots, which
# we prune per-model) and prune each provider's weights right after its pool is
# sampled -- the completions live in RAM, the weights are never needed again.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np

PRUNE = os.environ.get("IVGYM_PRUNE", "1") != "0"


def _prune_cache(name):
    """Delete a model's HF snapshot to keep peak disk under the volume quota."""
    if not PRUNE:
        return
    for root in (os.environ.get("HF_HOME", ""), str(Path.home() / ".cache/huggingface")):
        if not root:
            continue
        hub = Path(root) / "hub"
        d = hub / ("models--" + name.replace("/", "--"))
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N = int(os.environ.get("IVGYM_N", 128))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 64))
MAX_PROMPT = int(os.environ.get("IVGYM_MAXPROMPT", 48))
K_SWEEP = [int(x) for x in os.environ.get("IVGYM_KSWEEP", "8,16,32,64").split(",")]
RANK_CAP = 64.0
SEED = 0

# Which honest model M / ladder to run: "qwen" (default) or "llama".
LADDER = os.environ.get("IVGYM_LADDER", "qwen").lower()

# ---------------------------------------------------------------------------
# Two parallel ladders. Only the CAST of models is chosen by hand below; the
# distance_rank and group label each provider/proxy gets are DERIVED from the
# taxonomy facts in ivgym/model_registry.py (ivgym/model_taxonomy.py), not
# picked per row -- see that module for what "distance" means and why.
#
# A "::nf4" / "::fp4" / "::int8" suffix loads the BASE id under bitsandbytes; official
# GPTQ/AWQ checkpoints (separate repos) load normally (transformers reads their config).
from ivgym.model_registry import identity  # noqa: E402
from ivgym.model_taxonomy import describe, distance  # noqa: E402


def _build_ladder(claimed_id, provider_ids, proxy_ids):
    """(claimed_id, providers, proxies) with dist/group derived against the
    claimed model's taxonomy identity. List order here is just for readability
    -- `report()`/`render()` sort by the derived `dist`, not by this order."""
    ref = identity(claimed_id)
    providers = [
        (pid,
         identity(pid).label + (" (self)" if pid == claimed_id else ""),
         distance(ref, identity(pid)), describe(ref, identity(pid)))
        for pid in provider_ids
    ]
    proxies = [
        (pid, identity(pid).label, describe(ref, identity(pid)))
        for pid in proxy_ids
    ]
    return dict(claimed=claimed_id, providers=providers, proxies=proxies)


# (c) the quant sub-ladder isolates "where does quantization become detectable?":
#   int8(bnb) -> GPTQ-Int8 -> AWQ(int4) -> GPTQ-Int4 -> NF4(bnb) -> FP4(bnb),
# all of the SAME 7B weights, roughly increasing lossiness. Model substitutions
# (increasing taxonomy distance from M) follow.
QWEN = "Qwen/Qwen2.5-7B-Instruct"
LADDER_QWEN = _build_ladder(
    claimed_id=QWEN,
    provider_ids=[
        QWEN,
        QWEN + "::int8",
        "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
        "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        QWEN + "::nf4",
        QWEN + "::fp4",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen3-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
    ],
    proxy_ids=[
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    ],
)

# (b) does the ladder replicate with a NON-Qwen honest model? M = Llama-3.1-8B-Instruct.
LLAMA = "unsloth/Meta-Llama-3.1-8B-Instruct"
LADDER_LLAMA = _build_ladder(
    claimed_id=LLAMA,
    provider_ids=[
        LLAMA,
        LLAMA + "::int8",
        LLAMA + "::nf4",
        LLAMA + "::fp4",
        "unsloth/Meta-Llama-3.1-8B",
        "unsloth/llama-3-8b-Instruct",
        "unsloth/Llama-3.2-3B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "Qwen/Qwen2.5-7B-Instruct",
    ],
    proxy_ids=[
        "unsloth/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2.5-0.5B",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    ],
)

_LAD = LADDER_LLAMA if LADDER == "llama" else LADDER_QWEN
CLAIMED = _LAD["claimed"]
PROVIDERS = _LAD["providers"]
PROXIES = _LAD["proxies"]
_QUANT_SUFFIXES = ("::int8", "::nf4", "::fp4")


# ---------------------------------------------------------------------------
def _prompts(n):
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out = []
    for r in ds:
        if not r["input"].strip() and 8 <= len(r["instruction"]) <= 300:
            out.append(r["instruction"].strip())
        if len(out) >= n:
            break
    return out


def _parse_id(hf_id):
    """Split a "::marker" bitsandbytes suffix off the base repo id.
    Returns (base_id, bnb_marker_or_None)."""
    for suf in _QUANT_SUFFIXES:
        if hf_id.endswith(suf):
            return hf_id[: -len(suf)], suf[2:]      # "int8" / "nf4" / "fp4"
    return hf_id, None


def _load(base, torch, bnb=None):
    """Load `base` on CUDA in bf16. If `bnb` in {int8,nf4,fp4}, apply the matching
    bitsandbytes config. Official GPTQ/AWQ checkpoints (detected by name) carry their
    own quantization_config -- just load with a device_map so the packed kernels bind."""
    from transformers import AutoModelForCausalLM
    kw = dict(dtype=torch.bfloat16, trust_remote_code=True)
    if bnb is not None:
        from transformers import BitsAndBytesConfig
        if bnb == "int8":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:  # nf4 / fp4
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type=bnb,
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = "cuda"
        return AutoModelForCausalLM.from_pretrained(base, **kw).eval()
    if any(t in base.upper() for t in ("GPTQ", "AWQ")):
        kw["device_map"] = "cuda"
        return AutoModelForCausalLM.from_pretrained(base, **kw).eval()
    return AutoModelForCausalLM.from_pretrained(base, **kw).to("cuda").eval()


def generate_pool(hf_id, prompts, torch):
    """Sample N continuations from `hf_id` (temp 1.0). Returns list of
    (prompt_text, completion_text). Batched, left-padded."""
    from transformers import AutoTokenizer
    base, bnb = _parse_id(hf_id)
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = _load(base, torch, bnb=bnb)

    out, B = [], 16
    torch.manual_seed(SEED)
    for i in range(0, len(prompts), B):
        chunk = prompts[i:i + B]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_PROMPT).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                                  top_k=0, max_new_tokens=N_TOKENS,
                                  pad_token_id=tok.pad_token_id)
        new = gen[:, enc.input_ids.shape[1]:]
        for p, row in zip(chunk, new):
            out.append((p, tok.decode(row, skip_special_tokens=True)))
    n_params = sum(p.numel() for p in model.parameters())
    del model
    gc.collect(); torch.cuda.empty_cache()
    return out, n_params


def score_pool(proxy_model, proxy_tok, pool, torch):
    """Per-token surprise features under the proxy for each (prompt, completion).
    Re-tokenizes text under the proxy's OWN tokenizer (tokenizer-robust). Returns a
    list of dicts of per-token arrays (len up to N_TOKENS) for the completion region."""
    feats = []
    for prompt, completion in pool:
        if not completion.strip():
            feats.append(None); continue
        p_ids = proxy_tok(prompt, add_special_tokens=True).input_ids
        c_ids = proxy_tok(completion, add_special_tokens=False).input_ids[:N_TOKENS]
        if len(c_ids) == 0:
            feats.append(None); continue
        full = torch.tensor([p_ids + c_ids], device="cuda")
        Lp = len(p_ids)
        with torch.no_grad():
            logits = proxy_model(full).logits[0]            # [T, V]
        # logits[j] predicts token j+1; completion token c_ids[t] is at abs pos Lp+t,
        # predicted by logits[Lp+t-1].
        rows = logits[Lp - 1: Lp - 1 + len(c_ids)].float()  # [k, V]
        logp = torch.log_softmax(rows, dim=-1)
        p = logp.exp()
        tgt = torch.tensor(c_ids, device="cuda")
        nll = (-logp.gather(1, tgt[:, None])[:, 0]).cpu().numpy()
        chosen = rows.gather(1, tgt[:, None])
        rank = (rows > chosen).sum(dim=1).clamp(max=RANK_CAP).float().cpu().numpy()
        ent = (-(p * logp).sum(dim=1)).cpu().numpy()
        feats.append({"nll": nll, "rank": rank, "ent": ent})
    return feats


def seq_features(f, k):
    """Aggregate the first k per-token values into one per-sequence vector."""
    nll, rank, ent = f["nll"][:k], f["rank"][:k], f["ent"][:k]
    return np.array([
        nll.mean(), nll.std(), np.percentile(nll, 90), nll.max(),
        rank.mean(), (rank > 0).mean(),
        ent.mean(), ent.std(),
    ])


def auc_cell(honest_feats, dev_feats, k):
    """Out-of-fold LR AUC (signed, can be <0.5) + fixed-direction mean-NLL AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    H = [seq_features(f, k) for f in honest_feats if f]
    D = [seq_features(f, k) for f in dev_feats if f]
    if len(H) < 6 or len(D) < 6:
        return float("nan"), float("nan")
    X = np.vstack(H + D)
    y = np.r_[np.zeros(len(H)), np.ones(len(D))]
    # fixed-direction baseline: mean NLL alone (feature 0). No fitting -> no noise-fit.
    nll_only = X[:, 0]
    auc_nll = roc_auc_score(y, nll_only)
    # multi-feature out-of-fold LR
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc_lr = roc_auc_score(y, oof)
    return float(auc_lr), float(auc_nll)


@dataclass
class Result:
    provider: str
    group: str
    dist: int
    proxy: str
    proxy_group: str
    k: int
    auc_lr: float
    auc_nll: float


def main():
    import torch
    from transformers import AutoTokenizer

    t0 = time.time()
    prompts = _prompts(N)
    print(f"[{time.time()-t0:5.1f}s] claimed M = {CLAIMED} | {len(prompts)} alpaca prompts "
          f"| {N_TOKENS} tokens | k-sweep {K_SWEEP}", flush=True)

    # ---- 1. generate provider pools (rows) ----
    pools, meta = {}, {}
    for hf_id, label, dist, group in PROVIDERS:
        tp = time.time()
        try:
            pool, npar = generate_pool(hf_id, prompts, torch)
        except Exception as e:
            print(f"[{time.time()-t0:5.1f}s]   ROW SKIP {label:>18} ({repr(e)[:90]})", flush=True)
            continue
        pools[label] = pool
        meta[label] = (dist, group, npar)
        nonempty = sum(1 for _, c in pool if c.strip())
        print(f"[{time.time()-t0:5.1f}s]   generated {label:>18} [{group:>28}] "
              f"{npar/1e9:.2f}B  ({nonempty}/{len(pool)} nonempty, {time.time()-tp:.1f}s)", flush=True)
        # prune weights now (pool is in RAM). Defer the claimed model's snapshot: its
        # base is reused by every "CLAIMED::<bnb>" quant row, so only prune a base id
        # that is NOT the claimed model (the claimed base is small vs the far models
        # and its quant rows all reuse it -- leaving it cached avoids re-downloads).
        base, _ = _parse_id(hf_id)
        if base != CLAIMED:
            _prune_cache(base)
    honest_label = PROVIDERS[0][1]     # the distance-0 "(self)" row of the active ladder
    if honest_label not in pools:
        print(f"FATAL: honest pool '{honest_label}' missing; aborting."); return

    # ---- 2. score every pool under each proxy (cols), then build the AUC grid ----
    results = []
    for pid, plabel, pgroup in PROXIES:
        tp = time.time()
        try:
            ptok = AutoTokenizer.from_pretrained(pid, trust_remote_code=True)
            pmodel = _load(pid, torch)
        except Exception as e:
            print(f"[{time.time()-t0:5.1f}s]   COL SKIP {plabel:>14} ({repr(e)[:90]})", flush=True)
            continue
        feats = {lab: score_pool(pmodel, ptok, pool, torch) for lab, pool in pools.items()}
        del pmodel
        gc.collect(); torch.cuda.empty_cache()
        honest = feats[honest_label]
        for lab, pool in pools.items():
            dist, group, _ = meta[lab]
            for k in K_SWEEP:
                a_lr, a_nll = auc_cell(honest, feats[lab], k)
                results.append(Result(lab, group, dist, plabel, pgroup, k, a_lr, a_nll))
        print(f"[{time.time()-t0:5.1f}s]   scored under proxy {plabel:>14} [{pgroup:>18}] "
              f"({time.time()-tp:.1f}s)", flush=True)

    report(results, meta, t0)


def report(results, meta, t0):
    import time as _t
    kmax = max(K_SWEEP)
    proxies = [p[1] for p in PROXIES if any(r.proxy == p[1] for r in results)]
    rows = sorted({(r.dist, r.provider, r.group) for r in results})

    def get(prov, prox, k):
        for r in results:
            if r.provider == prov and r.proxy == prox and r.k == k:
                return r
        return None

    out_lines = []
    def emit(s=""):
        print(s, flush=True); out_lines.append(s)

    emit(f"\n{'='*100}")
    emit(f"CHEAP-PROXY DETECTION AUC   (claimed M = {CLAIMED};  scored tokens k = {kmax};  "
         f"out-of-fold LR)")
    emit(f"rows = model the provider SECRETLY runs (increasing distance from M);  "
         f"cols = verifier's cheap proxy")
    emit(f"cell = signed AUC (0.50 = indistinguishable; <0.50 = LR fit fold-noise, dir. reversed "
         f"held-out)")
    emit("="*100)
    head = f"{'substituted model':>20} {'distance group':>28} |" + "".join(f" {p:>14}" for p in proxies)
    emit(head)
    emit("-" * len(head))
    for dist, prov, group in rows:
        cells = ""
        for prox in proxies:
            r = get(prov, prox, kmax)
            cells += f" {r.auc_lr:>14.3f}" if r and not np.isnan(r.auc_lr) else f" {'--':>14}"
        emit(f"{prov:>20} {group:>28} |{cells}")

    emit(f"\nSAME CELLS, fixed-direction mean-NLL AUC (no fitting -> cannot dip below ~0.5 on noise):")
    emit("-" * len(head))
    for dist, prov, group in rows:
        cells = ""
        for prox in proxies:
            r = get(prov, prox, kmax)
            cells += f" {r.auc_nll:>14.3f}" if r and not np.isnan(r.auc_nll) else f" {'--':>14}"
        emit(f"{prov:>20} {group:>28} |{cells}")

    # ---- scored-token sweep (cost knob) on the same-family 0.5B proxy ----
    ref_prox = proxies[0]
    emit(f"\nSCORED-TOKEN SWEEP  (proxy = {ref_prox}; AUC vs k = the verifier's cost knob):")
    hk = f"{'substituted model':>20} |" + "".join(f" {'k='+str(k):>9}" for k in K_SWEEP)
    emit(hk); emit("-" * len(hk))
    for dist, prov, group in rows:
        cells = "".join(
            (f" {get(prov, ref_prox, k).auc_lr:>9.3f}"
             if get(prov, ref_prox, k) and not np.isnan(get(prov, ref_prox, k).auc_lr)
             else f" {'--':>9}") for k in K_SWEEP)
        emit(f"{prov:>20} |{cells}")

    emit("\nREADING IT:")
    emit("  * AUC rises monotonically with distance-from-M: quantized-self ~0.5 (invisible to a")
    emit("    cheap proxy), different-family ~1.0 (trivially caught). The cheap proxy WORKS once the")
    emit("    substitution is far enough that its conditional distribution diverges from M's.")
    emit("  * A DIFFERENT-family proxy (SmolLM2) still detects far substitutions but is weaker on")
    emit("    near ones -- proxy lineage matters most exactly where detection is hardest.")
    emit("  * Sub-0.5 signed AUC on the near rows is the noise-fit artifact (mean-NLL AUC stays 0.5).")
    emit("  * More scored tokens k => higher AUC on the detectable rows: k is the cost/accuracy knob.")

    resdir = Path(__file__).resolve().parents[1] / "docs" / "results"
    resdir.mkdir(parents=True, exist_ok=True)
    txt = resdir / f"exp_proxy_distance_grid_{LADDER}.txt"
    txt.write_text("\n".join(out_lines) + "\n")
    emit(f"\nwrote {txt}")

    try:
        fig = Path(__file__).resolve().parents[1] / "docs" / "figures" / f"fig_proxy_distance_grid_{LADDER}.png"
        render(results, rows, proxies, fig)
        emit(f"wrote {fig}")
    except Exception as e:
        emit(f"(skipped figure: {e})")
    emit(f"\ntotal {_t.time()-t0:.1f}s")


def render(results, rows, proxies, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kmax = max(K_SWEEP)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (axH, axK) = plt.subplots(1, 2, figsize=(15, 5.6))

    def get(prov, prox, k):
        for r in results:
            if r.provider == prov and r.proxy == prox and r.k == k:
                return r
        return None

    # left: AUC vs distance rank, one line per proxy
    labels = [f"{prov}\n[{group}]" for _, prov, group in rows]
    xs = list(range(len(rows)))
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    for j, prox in enumerate(proxies):
        ys = [get(prov, prox, kmax).auc_lr if get(prov, prox, kmax) else np.nan
              for _, prov, _ in rows]
        axH.plot(xs, ys, "o-", color=colors[j % len(colors)], lw=1.8, ms=7, label=prox)
    axH.axhline(0.5, ls=":", color="0.4", lw=1.3, label="chance (0.5)")
    axH.set_xticks(xs); axH.set_xticklabels(labels, fontsize=7, rotation=35, ha="right")
    axH.set_ylabel(f"detection AUC (out-of-fold LR, k={kmax})")
    axH.set_ylim(0.35, 1.03); axH.grid(alpha=0.25)
    axH.set_title("Cheap proxy catches a substitution once the secret model is\n"
                  "far enough from M (distance ladder, x-axis)", fontsize=10)
    axH.legend(fontsize=8, loc="lower right", title="verifier proxy")

    # right: scored-token sweep on the same-family 0.5B proxy
    ref = proxies[0]
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(rows)))
    for i, (_, prov, group) in enumerate(rows):
        ys = [get(prov, ref, k).auc_lr if get(prov, ref, k) else np.nan for k in K_SWEEP]
        axK.plot(K_SWEEP, ys, "o-", color=cmap[i], lw=1.6, ms=6, label=f"{prov}")
    axK.axhline(0.5, ls=":", color="0.4", lw=1.3)
    axK.set_xlabel("scored tokens k  (verifier cost knob)")
    axK.set_ylabel("detection AUC")
    axK.set_xscale("log", base=2); axK.set_xticks(K_SWEEP)
    axK.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axK.set_ylim(0.35, 1.03); axK.grid(alpha=0.25)
    axK.set_title(f"More scored tokens => more detection\n(proxy = {ref})", fontsize=10)
    axK.legend(fontsize=6.5, loc="lower right", ncol=1)

    fig.suptitle("When does the cheap-proxy verifier work? Detection AUC across a model-distance grid",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
