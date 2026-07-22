"""Within-family conditional-distribution correlation: the statistic that BOUNDS
every black-box proxy detector.

The black-box I/O detectors (`surface_stat`/`surface_rank`/`logit_judge` in
`ivgym/verifiers.py`) score a claimed token by its surprisal/rank under a small
*proxy* LM. They work AT ALL only because, within a model family, the proxy's
conditional next-token distribution agrees with the large reference M's. This
experiment measures that agreement directly on a real Qwen3 family ladder, and
shows the limit that follows from it:

    E[surface_stat | honest]  =  H(M, proxy)  =  H(M) + KL(M || proxy)

so the ONLY signal a proxy detector can ever see is KL(M||proxy) -- the
proxy<->M *disagreement*. The tighter the family correlation, the smaller that
budget, and the more an output-preserving attack (seed_43; a temp-retuned quant,
examples/seed_free_strategies.py) hides inside it. This is the quantitative reason
proxy detectors can never close the recompute-dominant gap in
`experiments/exp_io_detector_gpu.py`.

For each honest continuation SAMPLED FROM M, we re-score every position under M and
each proxy and compute, per position then averaged:

  top1_agree     P(argmax proxy == argmax M)                  (hard speculative-decode agreement)
  top8_jaccard   |top8(M) ∩ top8(proxy)| / |top8 union|
  spearman_topk  rank corr of the two logit vectors over M's top-K tokens
  accept_rate    E_{x~proxy}[min(1, p_M(x)/p_proxy(x))] = 1 - TV(M,proxy)
                 -- the EXACT expected speculative-decoding acceptance rate
  kl_m_proxy     KL(M || proxy) in nats  (= the surface_stat detector's blind-spot budget)
  surprisal_r    Pearson r between per-token -log p_M(tok) and -log p_proxy(tok)
                 (the literal correlation of the surface_stat signal between the two models)

NULL FLOOR (the control). The SAME proxy distributions paired with M distributions
from a DIFFERENT, shuffled position. This keeps both marginals identical but
destroys the *conditional* relationship, isolating "the proxy tracks M at THIS
position" from "both models just share unigram statistics." Every metric collapses
toward chance here. (A cross-FAMILY model would be the other natural control, but a
different family has a different tokenizer/vocab, so these token-aligned distribution
metrics are undefined across families; the shuffled-context null is the within-vocab
stand-in for "no conditional relationship.") The cross-FAMILY control IS now measured
directly in `experiments/exp_cross_family_accept.py`, which exploits the fact that
Qwen3 / Qwen2.5 / Qwen2.5-Coder / DeepSeek-R1-Distill-Qwen share Qwen's EXACT token
ids -- so accept-rate/top1/KL stay token-aligned across those families and show the
accept rate falling monotonically with family distance (see docs/SPEC_DECODING_AND_PROXY_DETECTION.md).

Run (validated on a single H100-80GB):
    /root/.venv/bin/python -m experiments.exp_family_correlation

Env overrides:
  IVGYM_M           reference (big) model           (default Qwen/Qwen3-4B)
  IVGYM_PROXIES     comma-sep proxy ladder          (default Qwen/Qwen3-1.7B,Qwen/Qwen3-0.6B)
  IVGYM_PROMPTS     prompts (default 10)
  IVGYM_TOKENS      sampled continuation length     (default 48)
  IVGYM_MAXPROMPT   prompt truncation in tokens     (default 32)
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym.backends.hf_gpu import DEFAULT_PROMPTS
from ivgym.model_registry import REGISTRY

# PROXIES is env-overridable to an arbitrary HF id, so registry lookups here
# are soft (fall back to "?") -- unlike the ladder experiments, this file
# doesn't require every model to have a taxonomy entry, only annotates when one exists.
def _family(hf_id):
    m = REGISTRY.get(hf_id)
    return m.family if m else "?"

M_NAME = os.environ.get("IVGYM_M", "Qwen/Qwen3-4B")
PROXIES = [s for s in os.environ.get(
    "IVGYM_PROXIES", "Qwen/Qwen3-1.7B,Qwen/Qwen3-0.6B").split(",") if s]
N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 10))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 48))
MAX_PROMPT = int(os.environ.get("IVGYM_MAXPROMPT", 32))
TOPK_OVERLAP = 8        # |top8 ∩|, mirrors verifiers' frac_in_top8
TOPK_SPEARMAN = 64      # rank corr over M's top-K, mirrors RANK_CAP
SHORT = {              # pretty/short labels for the figure axis
    "Qwen/Qwen3-0.6B": "0.6B", "Qwen/Qwen3-1.7B": "1.7B",
    "Qwen/Qwen3-4B": "4B", "Qwen/Qwen3-8B": "8B", "Qwen/Qwen3-14B": "14B",
}


@dataclass
class PairStats:
    name: str
    n_params: float
    top1_agree: float
    top8_jaccard: float
    spearman_topk: float
    accept_rate: float
    kl_m_proxy: float
    surprisal_r: float
    # null-floor counterparts (shuffled conditional)
    null_top1: float
    null_jaccard: float
    null_accept: float
    null_kl: float
    # per-token surprisals for the scatter figure
    m_surprisal: np.ndarray = field(default=None, repr=False)
    p_surprisal: np.ndarray = field(default=None, repr=False)


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = float((a.norm() * b.norm()))
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def _spearman_rows(m_vals, p_vals):
    """Mean per-row Spearman corr over the gathered top-K columns (torch [T,K])."""
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


def _logprobs_over(model, full_ids, L, n_tokens, torch):
    """Per-position log-softmax rows [n_tokens, V] predicting the continuation."""
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

    # --- sample honest continuations from M, then cache M's log-prob rows ---
    prompts = DEFAULT_PROMPTS[:N_PROMPTS]
    m_rows, claimed_rows = [], []
    torch.manual_seed(0)
    for pi, text in enumerate(prompts):
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
        # stash the full ids so each proxy re-scores the SAME sequence
        prompts[pi] = (text, full, L, n)
    M_lp = torch.cat(m_rows, dim=0)                 # [T, V]
    claimed = torch.cat(claimed_rows, dim=0)        # [T]
    T = M_lp.shape[0]
    del M, m_rows
    torch.cuda.empty_cache()
    print(f"  sampled + scored {T} honest tokens under M ({time.time()-t0:.1f}s)", flush=True)

    # one fixed shuffle for the conditional-null (no Math.random; seeded numpy)
    perm = torch.tensor(np.random.default_rng(0).permutation(T), device="cuda")
    M_top8 = M_lp.topk(TOPK_OVERLAP, dim=1).indices
    M_topk_idx = M_lp.topk(TOPK_SPEARMAN, dim=1).indices
    M_argmax = M_lp.argmax(dim=1)
    p_M = M_lp.exp()
    m_surp = (-M_lp.gather(1, claimed[:, None])[:, 0]).cpu().numpy()

    results = []
    for name in PROXIES:
        tp = time.time()
        P = _load(name, torch)
        if int(P.config.vocab_size) != V:
            raise ValueError(f"{name} vocab {P.config.vocab_size} != M vocab {V}; "
                             "token-aligned metrics need a shared tokenizer (same family).")
        pp = sum(p.numel() for p in P.parameters())
        rows = [_logprobs_over(P, full, L, n, torch) for (_, full, L, n) in prompts]
        P_lp = torch.cat(rows, dim=0)
        del P, rows
        torch.cuda.empty_cache()

        # ---- aligned (true conditional) metrics ----
        top1 = float((P_lp.argmax(dim=1) == M_argmax).float().mean())
        P_top8 = P_lp.topk(TOPK_OVERLAP, dim=1).indices
        # jaccard of the two top-8 sets, per row
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

        # ---- conditional-null (shuffle M relative to proxy) ----
        n_top1 = float((P_lp.argmax(dim=1) == M_argmax[perm]).float().mean())
        n_inter = torch.tensor([
            len(set(a.tolist()) & set(b.tolist())) for a, b in zip(M_top8[perm], P_top8)],
            device="cuda", dtype=torch.float32)
        n_jac = float((n_inter / (2 * TOPK_OVERLAP - n_inter)).mean())
        n_tv = 0.5 * (p_M[perm] - p_P).abs().sum(dim=1)
        n_accept = float((1.0 - n_tv).mean())
        n_kl = float((p_M[perm] * (M_lp[perm] - P_lp)).sum(dim=1).mean())

        results.append(PairStats(
            name=name, n_params=pp, top1_agree=top1, top8_jaccard=jac,
            spearman_topk=spear, accept_rate=accept, kl_m_proxy=kl, surprisal_r=surp_r,
            null_top1=n_top1, null_jaccard=n_jac, null_accept=n_accept, null_kl=n_kl,
            m_surprisal=m_surp, p_surprisal=p_surp))
        del P_lp, p_P
        torch.cuda.empty_cache()
        print(f"  {name:>16}: {pp/1e9:.2f}B  top1={top1:.3f} accept={accept:.3f} "
              f"KL={kl:.3f}  ({time.time()-tp:.1f}s)", flush=True)

    return m_params, V, T, results, time.time() - t0


def main():
    m_params, V, T, res, elapsed = run()

    print(f"\nWITHIN-FAMILY conditional agreement  (reference M = {M_NAME} "
          f"[family={_family(M_NAME)}], {m_params/1e9:.2f}B; {T} honest tokens)")
    print("Each proxy's CONDITIONAL next-token distribution vs M's, on tokens M actually sampled.\n"
          "'null' = same distributions, shuffled position (conditional relationship destroyed).\n")
    h = (f"{'proxy':>16} {'family':>8} {'params':>8} | {'top1':>6} {'top8_jac':>9} {'spear':>6} "
         f"{'accept':>7} {'KL(M||p)':>9} {'surp_r':>7} |  null: {'top1':>5} {'jac':>5} "
         f"{'accept':>7} {'KL':>6}")
    print(h + "\n" + "-" * len(h))
    for r in res:
        print(f"{SHORT.get(r.name, r.name):>16} {_family(r.name):>8} {r.n_params/1e9:>7.2f}B | "
              f"{r.top1_agree:>6.3f} {r.top8_jaccard:>9.3f} {r.spearman_topk:>6.3f} "
              f"{r.accept_rate:>7.3f} {r.kl_m_proxy:>9.3f} {r.surprisal_r:>7.3f} |  "
              f"      {r.null_top1:>5.3f} {r.null_jaccard:>5.3f} {r.null_accept:>7.3f} "
              f"{r.null_kl:>6.3f}")

    print("\nREADING IT:")
    print("  * top1/top8/accept/surp_r FAR above their null columns  => strong CONDITIONAL")
    print("    agreement (not just shared marginals). This overlap is what surface_stat exploits.")
    print("  * accept_rate = 1 - TV(M,proxy) is the exact expected speculative-decoding")
    print("    acceptance rate -- the cleanest single number for 'how much the proxy tracks M'.")
    print("  * KL(M||proxy) is small and SHRINKS as the proxy approaches M's size: it is the")
    print("    entire detection budget of any proxy detector (E[surface_stat]=H(M)+KL). An")
    print("    output-preserving attack (seed_43) moves 0 of it -> proxy detectors stay at floor,")
    print("    only recomputation of M (token_difr) separates it. See exp_io_detector_gpu.py.")

    try:
        out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "fig_family_correlation.png"
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
    fig, (axS, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- left: surprisal scatter for the SMALLEST proxy (largest size gap) ---
    small = min(res, key=lambda r: r.n_params)
    axS.hexbin(small.m_surprisal, small.p_surprisal, gridsize=40, cmap="viridis",
               bins="log", mincnt=1)
    lo = 0.0
    hi = float(max(small.m_surprisal.max(), small.p_surprisal.max()))
    axS.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="y = x")
    axS.set_xlabel(f"surprisal under M ({SHORT.get(m_name, m_name)})  −log p_M(token)  [nats]")
    axS.set_ylabel(f"surprisal under proxy ({SHORT.get(small.name)})  −log p_proxy(token)")
    axS.set_title(f"per-token surprisal tracks across the family\nPearson r = {small.surprisal_r:.3f}"
                  f"   (proxy = {SHORT.get(small.name)}, M = {SHORT.get(m_name, m_name)})", fontsize=10)
    axS.legend(loc="upper left", fontsize=9)
    axS.grid(alpha=0.2)

    # --- right: the LADDER -- agreement & KL vs proxy size (log-x), with the
    # conditional-null floor. Sorted small->large so the monotone trend reads L->R. ---
    rs = sorted(res, key=lambda r: r.n_params)
    xp = np.array([r.n_params / 1e9 for r in rs])
    axB.plot(xp, [r.accept_rate for r in rs], "o-", color="#2ca02c", lw=2, ms=8,
             label="accept rate (1−TV)")
    axB.plot(xp, [r.top1_agree for r in rs], "s-", color="#1f77b4", lw=2, ms=7,
             label="top-1 argmax agree")
    axB.plot(xp, [r.top8_jaccard for r in rs], "^-", color="#ff7f0e", lw=2, ms=7,
             label="top-8 Jaccard")
    # conditional-null floor (shuffled position) -- essentially 0 for every metric
    nullf = float(np.mean([r.null_accept for r in rs] + [r.null_top1 for r in rs]
                          + [r.null_jaccard for r in rs]))
    axB.axhline(nullf, ls=":", color="0.4", lw=1.4,
                label=f"shuffled-position null (≈{nullf:.02f})")
    axB.set_xscale("log")
    from matplotlib.ticker import NullLocator, NullFormatter
    axB.xaxis.set_minor_locator(NullLocator())
    axB.xaxis.set_minor_formatter(NullFormatter())
    axB.set_xticks(xp); axB.set_xticklabels([SHORT.get(r.name, r.name) for r in rs])
    axB.set_xlim(min(xp) * 0.8, max(xp) * 1.25)
    axB.set_xlabel("proxy size  (params, log scale;  M = "
                   f"{SHORT.get(m_name, m_name)})")
    axB.set_ylim(0, 1.0)
    axB.set_ylabel("conditional agreement with M")
    axB.grid(alpha=0.2)
    axB.legend(fontsize=8, loc="center left")

    # KL(M‖proxy) on a twin axis -- the detection budget, shrinking as proxy->M
    axK = axB.twinx()
    axK.plot(xp, [r.kl_m_proxy for r in rs], "D--", color="#d62728", lw=1.8, ms=6,
             label="KL(M‖proxy)  [nats]")
    axK.set_ylabel("KL(M‖proxy)  [nats]  — the proxy detector's entire budget",
                   color="#d62728")
    axK.tick_params(axis="y", labelcolor="#d62728")
    axK.set_ylim(0, max(r.kl_m_proxy for r in rs) * 1.4)
    axK.legend(fontsize=8, loc="center right")
    axB.set_title("agreement rises & KL shrinks as proxy → M;  null floor ≈ 0\n"
                  "(the gap above the floor is genuine conditional structure)", fontsize=10)

    fig.suptitle(f"Within-family conditional-distribution correlation (Qwen3 family, M={SHORT.get(m_name, m_name)})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
