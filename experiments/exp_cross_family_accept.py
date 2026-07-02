"""Does the proxy<->M agreement COLLAPSE across families? (the other half of the
speculative-decoding story).

`exp_family_correlation.py` measured *within-family* conditional agreement on a
Qwen3 size ladder and computed the exact expected speculative-decoding acceptance
rate `accept = E_{x~proxy}[min(1, p_M(x)/p_proxy(x))] = 1 - TV(M, proxy)` -- the
SAME quantity a real SD engine's rejection sampler realises (vLLM
`rejection_sampler.py`: accept iff `p_target/p_draft >= u`; SGLang
`tree_speculative_sampling_target_only`). That file explicitly PUNTED on the
cross-family control ("a different family has a different tokenizer/vocab, so
these token-aligned metrics are undefined across families").

This experiment removes that caveat. Every model below shares Qwen's *exact*
token ids (verified: Qwen3, Qwen2.5, Qwen2.5-Coder, and DeepSeek-R1-Distill-Qwen
all encode text to identical ids; all have vocab_size 151936). So we can measure
the token-aligned agreement metrics across a graded FAMILY-DISTANCE axis while
holding the tokenizer -- and largely the pretraining data -- fixed:

    same family   Qwen3-1.7B, Qwen3-0.6B         siblings of M (Qwen3-4B)
    cross gen     Qwen2.5-1.5B, Qwen2.5-0.5B     prior Qwen generation, same tokenizer
    cross domain  Qwen2.5-Coder-1.5B             Qwen2.5 base, code post-training
    cross post    DeepSeek-R1-Distill-Qwen-1.5B  Qwen2.5 base, RL/reasoning distill

The two questions this settles, that within-family numbers alone cannot:

  1. "high within a family and collapses across families" -- does accept_rate /
     top1 drop for a cross-family proxy of the SAME SIZE as a same-family one?
  2. "is it just low perplexity from shared training data?" -- the cross-domain /
     cross-post proxies are literally Qwen bases (they SHARE M's pretraining data)
     yet differ in post-training. If agreement tracks family-distance rather than
     data overlap, the signal is CONDITIONAL-distribution agreement, not generic
     fluency / shared-corpus perplexity.

Matched-size blocks (Qwen3-1.7B vs the three ~1.5B cross-family models; Qwen3-0.6B
vs Qwen2.5-0.5B) isolate FAMILY from SIZE. The shuffled-position null (same
marginals, conditional relationship destroyed) is the floor every metric would
collapse to with no conditional structure at all.

Run (single H100-80GB; downloads ~23GB, pruned as it goes so peak disk stays low):
    IVGYM_PRUNE=1 /home/ubuntu/inference-verification/.venv/bin/python \
        -m experiments.exp_cross_family_accept

Env overrides:
  IVGYM_M         reference (big) model      (default Qwen/Qwen3-4B)
  IVGYM_PROMPTS   prompts (default 16)
  IVGYM_TOKENS    continuation length        (default 64)
  IVGYM_MAXPROMPT prompt truncation tokens   (default 32)
  IVGYM_PRUNE     1 => delete each proxy's HF cache after scoring (default 1)
"""
from __future__ import annotations

import gc
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym.backends.hf_gpu import DEFAULT_PROMPTS

M_NAME = os.environ.get("IVGYM_M", "Qwen/Qwen3-4B")
N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 16))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 64))
MAX_PROMPT = int(os.environ.get("IVGYM_MAXPROMPT", 32))
PRUNE = os.environ.get("IVGYM_PRUNE", "1") != "0"

TOPK_OVERLAP = 8
TOPK_SPEARMAN = 64

# (hf id, short label, family-distance group). Order = increasing distance from M.
PROXIES = [
    ("Qwen/Qwen3-1.7B",                       "Qwen3-1.7B",   "same family"),
    ("Qwen/Qwen3-0.6B",                       "Qwen3-0.6B",   "same family"),
    ("Qwen/Qwen2.5-1.5B",                     "Qwen2.5-1.5B", "cross gen"),
    ("Qwen/Qwen2.5-0.5B",                     "Qwen2.5-0.5B", "cross gen"),
    ("Qwen/Qwen2.5-Coder-1.5B",               "Qwen2.5-Coder-1.5B", "cross domain"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DS-R1-Qwen-1.5B", "cross post"),
]
GROUP_ORDER = ["same family", "cross gen", "cross domain", "cross post"]


@dataclass
class PairStats:
    name: str
    label: str
    group: str
    n_params: float
    top1_agree: float
    top8_jaccard: float
    spearman_topk: float
    accept_rate: float
    kl_m_proxy: float
    surprisal_r: float
    null_top1: float
    null_jaccard: float
    null_accept: float
    m_surprisal: np.ndarray = field(default=None, repr=False)
    p_surprisal: np.ndarray = field(default=None, repr=False)


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = float((a.norm() * b.norm()))
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def _spearman_rows(m_vals, p_vals):
    import torch
    def ranks(x):
        return x.argsort(dim=1).argsort(dim=1).float()
    rm, rp = ranks(m_vals), ranks(p_vals)
    rm = rm - rm.mean(dim=1, keepdim=True)
    rp = rp - rp.mean(dim=1, keepdim=True)
    num = (rm * rp).sum(dim=1)
    den = rm.norm(dim=1) * rp.norm(dim=1) + 1e-12
    return float((num / den).mean())


def _load(name, torch):
    from transformers import AutoModelForCausalLM
    return (AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval())


def _prune_cache(name):
    """Delete a model's HF cache dir to keep peak disk low (23GB of proxies on a
    30GB disk). M is never pruned."""
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    d = cache / ("models--" + name.replace("/", "--"))
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def _logprobs_over(model, full_ids, L, n_tokens, torch):
    with torch.no_grad():
        logits = model(full_ids).logits[0, L - 1: L - 1 + n_tokens]
    return torch.log_softmax(logits.float(), dim=-1)


def run():
    import torch
    from transformers import AutoTokenizer

    t0 = time.time()
    print(f"loading M = {M_NAME} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(M_NAME)
    M = _load(M_NAME, torch)
    V = int(M.config.vocab_size)
    m_params = sum(p.numel() for p in M.parameters())
    print(f"  M loaded ({time.time()-t0:.1f}s) vocab={V} params={m_params/1e9:.2f}B", flush=True)

    # --- sample honest continuations from M (full distribution: temp 1, no top-k/p) ---
    prompts = DEFAULT_PROMPTS[:N_PROMPTS]
    m_rows, claimed_rows, stashed = [], [], []
    torch.manual_seed(0)
    for text in prompts:
        ids = tok(text, return_tensors="pt").input_ids[:, :MAX_PROMPT].to("cuda")
        L = ids.shape[1]
        with torch.no_grad():
            gen = M.generate(ids, do_sample=True, temperature=1.0, top_p=1.0,
                             top_k=0, max_new_tokens=N_TOKENS,
                             pad_token_id=tok.eos_token_id)
        full = gen[:, : L + N_TOKENS]
        n = full.shape[1] - L
        m_rows.append(_logprobs_over(M, full, L, n, torch))
        claimed_rows.append(full[0, L: L + n])
        stashed.append((full, L, n))
    M_lp = torch.cat(m_rows, dim=0)
    claimed = torch.cat(claimed_rows, dim=0)
    T = M_lp.shape[0]
    del M, m_rows
    gc.collect(); torch.cuda.empty_cache()
    print(f"  sampled + scored {T} honest tokens under M ({time.time()-t0:.1f}s)", flush=True)

    perm = torch.tensor(np.random.default_rng(0).permutation(T), device="cuda")
    M_top8 = M_lp.topk(TOPK_OVERLAP, dim=1).indices
    M_topk_idx = M_lp.topk(TOPK_SPEARMAN, dim=1).indices
    M_argmax = M_lp.argmax(dim=1)
    p_M = M_lp.exp()
    m_surp = (-M_lp.gather(1, claimed[:, None])[:, 0]).cpu().numpy()

    results = []
    for name, label, group in PROXIES:
        tp = time.time()
        try:
            P = _load(name, torch)
        except Exception as e:
            print(f"  {label:>20}: LOAD FAILED ({repr(e)[:80]}) -- skipping", flush=True)
            continue
        if int(P.config.vocab_size) != V:
            print(f"  {label:>20}: vocab {P.config.vocab_size} != {V}; skipping", flush=True)
            del P; torch.cuda.empty_cache(); continue
        pp = sum(p.numel() for p in P.parameters())
        rows = [_logprobs_over(P, full, L, n, torch) for (full, L, n) in stashed]
        P_lp = torch.cat(rows, dim=0)
        del P, rows
        gc.collect(); torch.cuda.empty_cache()

        top1 = float((P_lp.argmax(dim=1) == M_argmax).float().mean())
        P_top8 = P_lp.topk(TOPK_OVERLAP, dim=1).indices
        inter = torch.tensor([
            len(set(a.tolist()) & set(b.tolist())) for a, b in zip(M_top8, P_top8)],
            device="cuda", dtype=torch.float32)
        jac = float((inter / (2 * TOPK_OVERLAP - inter)).mean())
        spear = _spearman_rows(M_lp.gather(1, M_topk_idx), P_lp.gather(1, M_topk_idx))
        p_P = P_lp.exp()
        tv = 0.5 * (p_M - p_P).abs().sum(dim=1)
        accept = float((1.0 - tv).mean())
        kl = float((p_M * (M_lp - P_lp)).sum(dim=1).mean())
        p_surp = (-P_lp.gather(1, claimed[:, None])[:, 0]).cpu().numpy()
        surp_r = _pearson(torch.tensor(m_surp), torch.tensor(p_surp))

        n_top1 = float((P_lp.argmax(dim=1) == M_argmax[perm]).float().mean())
        n_inter = torch.tensor([
            len(set(a.tolist()) & set(b.tolist())) for a, b in zip(M_top8[perm], P_top8)],
            device="cuda", dtype=torch.float32)
        n_jac = float((n_inter / (2 * TOPK_OVERLAP - n_inter)).mean())
        n_tv = 0.5 * (p_M[perm] - p_P).abs().sum(dim=1)
        n_accept = float((1.0 - n_tv).mean())

        results.append(PairStats(
            name=name, label=label, group=group, n_params=pp, top1_agree=top1,
            top8_jaccard=jac, spearman_topk=spear, accept_rate=accept, kl_m_proxy=kl,
            surprisal_r=surp_r, null_top1=n_top1, null_jaccard=n_jac, null_accept=n_accept,
            m_surprisal=m_surp, p_surprisal=p_surp))
        del P_lp, p_P
        gc.collect(); torch.cuda.empty_cache()
        if PRUNE:
            _prune_cache(name)
        print(f"  {label:>20} [{group:>12}]: {pp/1e9:.2f}B  top1={top1:.3f} "
              f"accept={accept:.3f} KL={kl:.3f}  ({time.time()-tp:.1f}s)", flush=True)

    return m_params, V, T, results, time.time() - t0


def main():
    m_params, V, T, res, elapsed = run()
    if not res:
        print("no proxies scored"); return

    print(f"\nCROSS-FAMILY conditional agreement  (reference M = {M_NAME}, "
          f"{m_params/1e9:.2f}B; {T} honest tokens; shared Qwen tokenizer)")
    print("Every proxy shares M's EXACT token ids, so accept_rate/top1/KL are token-aligned\n"
          "across families. accept_rate = 1-TV(M,proxy) = the exact expected speculative-\n"
          "decoding acceptance rate (vLLM/SGLang rejection sampler). 'null' = shuffled position.\n")
    h = (f"{'proxy':>20} {'group':>13} {'params':>8} | {'top1':>6} {'top8':>6} "
         f"{'spear':>6} {'accept':>7} {'KL':>6} {'surp_r':>7} | null: {'top1':>5} {'acc':>5}")
    print(h + "\n" + "-" * len(h))
    for r in sorted(res, key=lambda r: (GROUP_ORDER.index(r.group), -r.n_params)):
        print(f"{r.label:>20} {r.group:>13} {r.n_params/1e9:>7.2f}B | "
              f"{r.top1_agree:>6.3f} {r.top8_jaccard:>6.3f} {r.spearman_topk:>6.3f} "
              f"{r.accept_rate:>7.3f} {r.kl_m_proxy:>6.3f} {r.surprisal_r:>7.3f} | "
              f"      {r.null_top1:>5.3f} {r.null_accept:>5.3f}")

    # ---- matched-size analysis: family vs size ----
    print("\nMATCHED-SIZE (isolates FAMILY from SIZE -- same size, same tokenizer, "
          "overlapping\npretraining data; only the model family/training differs):")
    by = {r.label: r for r in res}
    def block(title, same_label, cross_labels):
        s = by.get(same_label)
        if not s:
            return
        print(f"  {title}")
        print(f"    {s.label:>20} [same family]  accept={s.accept_rate:.3f}  top1={s.top1_agree:.3f}  KL={s.kl_m_proxy:.3f}")
        for cl in cross_labels:
            c = by.get(cl)
            if not c:
                continue
            dacc = c.accept_rate - s.accept_rate
            dtop = c.top1_agree - s.top1_agree
            print(f"    {c.label:>20} [{c.group:>12}]  accept={c.accept_rate:.3f} "
                  f"({dacc:+.3f})  top1={c.top1_agree:.3f} ({dtop:+.3f})  KL={c.kl_m_proxy:.3f}")
    block("~1.5-1.7B tier:", "Qwen3-1.7B",
          ["Qwen2.5-1.5B", "Qwen2.5-Coder-1.5B", "DS-R1-Qwen-1.5B"])
    block("~0.5-0.6B tier:", "Qwen3-0.6B", ["Qwen2.5-0.5B"])

    print("\nREADING IT:")
    print("  * If the same-family proxy has higher accept_rate/top1 than a cross-family proxy")
    print("    of the SAME SIZE, the agreement is FAMILY-specific -- exactly the speculative-")
    print("    decoding intuition ('accept rate high within a family, collapses across families').")
    print("  * The cross-domain/cross-post proxies are Qwen BASES (they share M's pretraining")
    print("    data) yet agree less -> the signal is CONDITIONAL-distribution agreement, NOT")
    print("    generic fluency or shared-corpus perplexity.")
    print("  * All proxies still sit ABOVE the shuffled-position null (they share a language),")
    print("    but the accept_rate is graded by family distance -- that grading is the point.")

    try:
        out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "fig_cross_family_accept.png"
        render(res, out, M_NAME)
        print(f"\nwrote figure: {out}")
    except Exception as e:
        print(f"\n(skipped figure: {e})")
    print(f"\ntotal {elapsed:.1f}s")


def render(res, path: Path, m_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (axB, axS) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # --- left: accept_rate vs size, colored by family group; null floor line ---
    colors = {"same family": "#2ca02c", "cross gen": "#ff7f0e",
              "cross domain": "#9467bd", "cross post": "#d62728"}
    markers = {"same family": "o", "cross gen": "s", "cross domain": "^", "cross post": "D"}
    for g in GROUP_ORDER:
        pts = sorted([r for r in res if r.group == g], key=lambda r: r.n_params)
        if not pts:
            continue
        axB.plot([r.n_params / 1e9 for r in pts], [r.accept_rate for r in pts],
                 markers[g] + "-", color=colors[g], ms=9, lw=1.6, label=g)
    nullf = float(np.mean([r.null_accept for r in res]))
    axB.axhline(nullf, ls=":", color="0.4", lw=1.4, label=f"shuffled null (≈{nullf:.02f})")
    for r in res:
        axB.annotate(r.label.replace("Qwen", "Q").replace("2.5", "2.5-"),
                     (r.n_params / 1e9, r.accept_rate), fontsize=6.5,
                     textcoords="offset points", xytext=(4, 4))
    axB.set_xscale("log")
    axB.set_xlabel("proxy size (params, log)")
    axB.set_ylabel("speculative-decoding accept rate  (1 − TV(M, proxy))")
    axB.set_ylim(0, 1.0)
    axB.grid(alpha=0.2)
    axB.legend(fontsize=8, loc="lower right", title=f"family distance from M={m_name.split('/')[-1]}")
    axB.set_title("Accept rate is graded by FAMILY DISTANCE, not size alone\n"
                  "(same tokenizer & pretraining data; only training differs)", fontsize=10)

    # --- right: matched-size bar chart at the ~1.5B tier ---
    tier = [r for r in res if 1.0e9 <= r.n_params <= 2.0e9]
    tier = sorted(tier, key=lambda r: -r.accept_rate)
    if tier:
        xs = np.arange(len(tier))
        axS.bar(xs, [r.accept_rate for r in tier],
                color=[colors[r.group] for r in tier])
        axS.axhline(float(np.mean([r.null_accept for r in tier])), ls=":", color="0.4",
                    lw=1.3, label="shuffled null")
        axS.set_xticks(xs)
        axS.set_xticklabels([f"{r.label}\n[{r.group}]" for r in tier], fontsize=7.5, rotation=0)
        axS.set_ylabel("accept rate (1 − TV)")
        axS.set_ylim(0, 1.0)
        for x, r in zip(xs, tier):
            axS.text(x, r.accept_rate + 0.01, f"{r.accept_rate:.3f}", ha="center", fontsize=8)
        axS.set_title("Matched ~1.5B size: same-family proxy wins\n"
                      "(size/tokenizer/pretraining held fixed)", fontsize=10)
        axS.legend(fontsize=8)
        axS.grid(alpha=0.2, axis="y")

    fig.suptitle("Cross-family speculative-decoding acceptance: high within family, "
                 "lower across families", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
