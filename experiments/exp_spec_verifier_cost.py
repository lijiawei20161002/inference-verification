"""SpeculativeVerifier on a GPU: **cost saving vs detection performance**.

This is the end-to-end, real-model run of the client-side speculative verifier
(``ivgym.spec_decode.ProxySpecVerifier``). It answers the one question the CPU
sweep and the older ``exp_proxy_spec_verify`` GPU section left open: *what does
the cheap proxy actually save, in real GPU cost, and what detection performance
do you give up for it?*

The setup (all on one CUDA host)
--------------------------------
* **Reference model M** (e.g. ``Qwen/Qwen3-4B``) -- the model the provider claims
  to run. The strong verifier ``defenses.token_difr`` RE-RUNS M for every claimed
  token: correct (AUC ~ 1.0) but as expensive as inference itself.
* **Proxy q** (e.g. ``Qwen/Qwen3-0.6B``, same tokenizer) -- a genuinely smaller,
  different model the *client* owns. The **SpeculativeVerifier** scores the
  speculative-decoding acceptance rate ``accept_rate = 1 − TV(p, q)`` between the
  provider-served distribution ``p`` (``backend.served_logits`` -- returned by a
  logprob API, free to the client) and its own proxy ``q``. It never runs M.

Scoring (matches ``spec_decode.ProxyReference``)
------------------------------------------------
A forward-pass cheat can move ``TV(p, q)`` in *either* direction, so the signal is
the **absolute deviation from the honest anchor**, ``|TV(p,q) − μ| / σ`` with
``μ, σ`` calibrated on honest tokens -- exactly what ``ProxyReference.score``
computes, here per token so it flows through the SAME winsorize / batch_means /
evaluate pipeline as ``token_difr`` (apples-to-apples AUC). The honest-null floor
is a same-distribution split of the honest tokens (→ AUC ~ 0.5).

What we report, side by side
----------------------------
  performance  : detection AUC (honest vs attack), batched like token_difr
  cost         : MEASURED GPU wall-clock per verified sequence (proxy prefill vs
                 full-M prefill) + the analytic FLOP ratio (= param ratio) + a
                 $/1M-verified-tokens figure at a configurable GPU price.

Reading the result (see ``docs/results/exp_spec_verifier_cost.txt``): the
SpeculativeVerifier costs ``params(q)/params(M)`` of the recompute (a large
saving), but at *realistic* forward-pass strength (quant_4bit / adv_quant_temp)
the acceptance rate barely moves -- the measured spec_accept AUC sits near
chance while ``token_difr`` (recompute) separates every attack. Real quant
noise moves ``TV(p, q)`` far less than a real model's honest run-to-run
variance, so this regime is RECOMPUTE-DOMINANT, exactly the boundary the CPU
sweep (``exp_proxy_spec_verify``) predicts for small sigma. The regime where
the cheap verifier *wins* is model substitution -- a wholesale change of the
served conditional distribution -- measured in the companion experiment
``exp_spec_substitution_gpu``. So the proxy **shrinks how often the exact
recompute must fire; it does not replace it.**

Run:
    IVGYM_M=Qwen/Qwen3-4B IVGYM_PROXY=Qwen/Qwen3-0.6B \
        .venv/bin/python -m experiments.exp_spec_verifier_cost

Env:  IVGYM_M, IVGYM_PROXY, IVGYM_PROMPTS, IVGYM_TOKENS, IVGYM_BATCH,
      IVGYM_ATTACKS (comma list), IVGYM_GPU_USD_PER_HR (default 2.50),
      IVGYM_FIG (output path for the Pareto figure).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, defenses, harness, spec_decode as sd
from ivgym.attacks import Quantization
from ivgym.core import SamplingSpec

M_NAME = os.environ.get("IVGYM_M", "Qwen/Qwen3-4B")
PROXY_NAME = os.environ.get("IVGYM_PROXY", "Qwen/Qwen3-0.6B")
N_PROMPTS = int(os.environ.get("IVGYM_PROMPTS", 24))
N_TOKENS = int(os.environ.get("IVGYM_TOKENS", 128))
BATCH = int(os.environ.get("IVGYM_BATCH", 150))
AUC_SEEDS = int(os.environ.get("IVGYM_AUC_SEEDS", 16))   # avg AUC over batch-composition seeds
USD_PER_HR = float(os.environ.get("IVGYM_GPU_USD_PER_HR", 2.50))
FIG = os.environ.get("IVGYM_FIG", "docs/figures/fig_spec_verifier_cost.png")

# A STRONG forward-pass corruption, to sit alongside the realistic-strength
# quant_4bit: it exposes the regime where the cheap proxy's accept rate DOES
# separate (large TV(p,q) shift), vs quant_4bit where only recompute holds.
attacks.register(Quantization(name="quant_strong", extra_sigma=0.7, bias_sigma=0.25, act_sigma=0.7))

ATTACKS = tuple(os.environ.get(
    "IVGYM_ATTACKS", "quant_strong,quant_4bit,adv_quant_temp,seed_43").split(","))


# ---------------------------------------------------------------------------
# Per-token raw signals (read served p and proxy q; M's logits NOWHERE).
# ---------------------------------------------------------------------------
def spec_accept_tv(backend, sequences):
    """Per-token ``TV(p, q)`` between the provider-SERVED distribution ``p``
    (``served_logits`` -- free from a logprob API) and the client PROXY ``q``
    (``proxy_logits`` -- one cheap forward pass). ``TV = 1 − accept_rate``: the
    draft-anchored speculative-decoding signal, made token-level."""
    out = []
    for seq in sequences:
        for s in seq.steps:
            p = sd.softmax(backend.served_logits(seq.prompt_id, s.position))
            q = sd.softmax(backend.proxy_logits(seq.prompt_id, s.position))
            out.append(sd.tv(p, q))
    return np.asarray(out, float)


def proxy_nll(backend, sequences):
    """Context baseline: proxy surprisal ``−log q(x)`` of the served tokens -- the
    token-only ``surface_stat`` cousin (same proxy pass, no extra cost). This is
    the entropy/cross-entropy-style fingerprint a temperature retune can hide
    behind (adv_quant_temp), unlike the draft-anchored accept rate."""
    out = []
    for seq in sequences:
        for s in seq.steps:
            lq = sd.log_softmax(backend.proxy_logits(seq.prompt_id, s.position))
            out.append(float(-lq[s.claimed_token]))
    return np.asarray(out, float)


def anchor_dev(raw, mu, sigma):
    """Standardized absolute deviation from the honest anchor, ``|x − μ| / σ`` --
    exactly ``ProxyReference.score``. A cheat that moves the statistic in EITHER
    direction becomes a high (anomalous) score; honest stays near 0."""
    return np.abs(raw - mu) / sigma


# ---------------------------------------------------------------------------
# The actual ProxySpecVerifier class, run end-to-end on real samples.
# ---------------------------------------------------------------------------
def build_proxy_samples(backend, sequences):
    """Wrap each served sequence as a ``spec_decode.ProxySample`` the real
    ``ProxySpecVerifier`` consumes: target = log-softmax of served logits ``p``,
    proxy = log-softmax of client proxy ``q``, plus the claimed tokens; ``_truth``
    = M's true logits (for the ``recompute_divergence`` reference line)."""
    out = []
    for seq in sequences:
        served = [sd.log_softmax(backend.served_logits(seq.prompt_id, s.position)) for s in seq.steps]
        proxy = [sd.log_softmax(backend.proxy_logits(seq.prompt_id, s.position)) for s in seq.steps]
        truth = [sd.log_softmax(backend.reference_logits(seq.prompt_id, s.position)) for s in seq.steps]
        out.append(sd.ProxySample(
            provider_name=seq.config_name, spec=SamplingSpec(),
            positions=[sd.Position(t, q) for t, q in zip(served, proxy)],
            served_tokens=[s.claimed_token for s in seq.steps], _truth=truth))
    return out


def batched_auc(honest_scores, attack_scores, name):
    """Batched AUC / TPR@1%FPR, AVERAGED over ``AUC_SEEDS`` train/test-split and
    batch-composition seeds. Averaging is essential, not cosmetic: a *single*
    ``harness.evaluate`` split draws the honest-test and attack-test tokens as two
    finite subsets whose sample means differ by chance; ``batch_means`` then
    concentrates each batch mean around its subset mean (~1/sqrt(BATCH)), which
    amplifies that accidental gap into an AUC far from the truth (identical
    distributions can score 0.38 / 0.63 on one seed). Averaging over seeds
    re-splits each time, so the accidental gaps cancel and the honest-null floor
    lands at ~0.5. See docs; regression-tested in tests/test_proxy_spec.py."""
    ts_h = harness.TokenScores("honest", {name: honest_scores})
    ts_a = harness.TokenScores("attack", {name: attack_scores})
    shim = type("D", (), {"name": name})()
    aucs, tprs = [], []
    for seed in range(AUC_SEEDS):
        r = harness.evaluate(ts_h, ts_a, [shim], [BATCH], seed=seed)[0]
        aucs.append(r.auc)
        tprs.append(r.tpr_at_1pct)
    return float(np.mean(aucs)), float(np.mean(tprs))


def split_floor(scores, name):
    """Same-distribution honest-null floor: AUC between two random halves of the
    honest scored tokens (→ ~0.5)."""
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(scores))
    half = len(scores) // 2
    return batched_auc(scores[perm[:half]], scores[perm[half:]], name)


def main():
    from ivgym.backends.hf_gpu import HFGPUBackend

    print("=" * 84)
    print("SpeculativeVerifier on GPU: cost saving vs detection performance")
    print("=" * 84)
    t0 = time.time()
    backend = HFGPUBackend(model_name=M_NAME, proxy_model_name=PROXY_NAME)
    ratio = backend.n_params / backend.proxy_n_params
    print(f"loaded ({time.time()-t0:.1f}s)   M={M_NAME} ({backend.n_params/1e9:.2f}B)   "
          f"proxy={PROXY_NAME} ({backend.proxy_n_params/1e9:.2f}B)   M/proxy = {ratio:.1f}x params")
    print(f"config: {N_PROMPTS} prompts x {N_TOKENS} tokens, batch={BATCH}\n")

    spec = SamplingSpec()
    td = defenses.get("token_difr")

    # --- honest reference set (calibration anchor; same prompts as attacks) ---
    print("generating honest reference set ...")
    honest = harness.generate_dataset(backend, attacks.get("honest"), spec,
                                       N_PROMPTS, N_TOKENS, record_activations=False)
    h_td = harness.verify(backend, honest, spec, [td]).scores["token_difr"]
    h_tv_raw = spec_accept_tv(backend, honest)
    h_nll_raw = proxy_nll(backend, honest)
    mu_tv, sd_tv = h_tv_raw.mean(), h_tv_raw.std() + 1e-12
    mu_nll, sd_nll = h_nll_raw.mean(), h_nll_raw.std() + 1e-12
    # honest-anchor scored arrays (deviation from own mean → centered near 0)
    h_spec = anchor_dev(h_tv_raw, mu_tv, sd_tv)
    h_nll = anchor_dev(h_nll_raw, mu_nll, sd_nll)
    print(f"honest anchor: accept_rate = 1 - TV(p,q) = {1.0 - mu_tv:.3f}  "
          f"(mean over served tokens)\n")

    # ------------------------------------------------------------------ AUCs
    print("DETECTION PERFORMANCE  (honest vs attack, batched AUC / TPR@1%FPR)")
    print(f"  {'config':>16} | {'spec_accept (SpecVerifier)':>27} | "
          f"{'proxy_nll (token-only)':>23} | {'token_difr (recompute)':>23}")
    print("  " + "-" * 96)

    # honest-null floor (same-distribution split)
    a_spec, t_spec = split_floor(h_spec, "spec_accept")
    a_nll, t_nll = split_floor(h_nll, "proxy_nll")
    a_td, t_td = split_floor(h_td, "token_difr")
    rows = [("(honest null)", a_spec, a_nll, a_td)]
    print(f"  {'(honest null)':>16} | {a_spec:>10.3f} (TPR {t_spec:>5.2f})     | "
          f"{a_nll:>8.3f} (TPR {t_nll:>4.2f}) | {a_td:>8.3f} (TPR {t_td:>4.2f})")

    attack_seqs = {}
    for name in ATTACKS:
        seqs = harness.generate_dataset(backend, attacks.get(name), spec,
                                        N_PROMPTS, N_TOKENS, record_activations=False)
        attack_seqs[name] = seqs
        a_td_raw = harness.verify(backend, seqs, spec, [td]).scores["token_difr"]
        a_spec = anchor_dev(spec_accept_tv(backend, seqs), mu_tv, sd_tv)
        a_nll = anchor_dev(proxy_nll(backend, seqs), mu_nll, sd_nll)
        auc_spec, tpr_spec = batched_auc(h_spec, a_spec, "spec_accept")
        auc_nll, tpr_nll = batched_auc(h_nll, a_nll, "proxy_nll")
        auc_td, tpr_td = batched_auc(h_td, a_td_raw, "token_difr")
        rows.append((name, auc_spec, auc_nll, auc_td))
        print(f"  {name:>16} | {auc_spec:>10.3f} (TPR {tpr_spec:>5.2f})     | "
              f"{auc_nll:>8.3f} (TPR {tpr_nll:>4.2f}) | {auc_td:>8.3f} (TPR {tpr_td:>4.2f})")

    # -------------------------------------------------------- the real class
    # Per-SEQUENCE verdicts: with only N_PROMPTS honest sequences a single
    # calibrate/held-out split is far too small (a 12/12 split gives a 90th-pct
    # threshold off ~12 z-scores -> held-out FPR swings 0.0-0.5 by luck). Average
    # the FPR and per-attack flag rate over AUC_SEEDS independent cal/null splits.
    print("\nProxySpecVerifier end-to-end (the class, per-sequence verdicts):")
    hs = build_proxy_samples(backend, honest)
    attack_samples = {name: build_proxy_samples(backend, attack_seqs[name])
                      for name in ATTACKS}
    cut = len(hs) // 2
    thr_acc, null_acc = [], []
    flag_acc = {name: [] for name in ATTACKS}
    for seed in range(AUC_SEEDS):
        perm = np.random.default_rng(seed).permutation(len(hs))
        cal = [hs[i] for i in perm[:cut]]
        null = [hs[i] for i in perm[cut:]]
        verifier = sd.ProxySpecVerifier(feature="accept_rate").calibrate(cal, fpr=0.10)
        thr_acc.append(verifier.threshold)
        null_acc.append(np.mean([verifier.verify(s).flagged for s in null]))
        for name in ATTACKS:
            flag_acc[name].append(np.mean([verifier.verify(s).flagged
                                           for s in attack_samples[name]]))
    print(f"  calibrated threshold z>{np.mean(thr_acc):.3f} (target FPR 0.10); "
          f"held-out honest FPR = {np.mean(null_acc):.2f} "
          f"(mean over {AUC_SEEDS} cal/null splits, N={len(hs)} honest seqs)")
    for name in ATTACKS:
        print(f"    {name:>16}: flagged {np.mean(flag_acc[name])*100:>5.1f}% "
              f"of served sequences")

    # ------------------------------------------------------------------ cost
    print("\n" + "=" * 84)
    print("COST  (measured GPU wall-clock, one prefill over [prompt+claimed] per seq)")
    print("=" * 84)
    ts, tc = backend.timed_seconds, backend.timed_calls
    ref_s = ts["reference"] / max(tc["reference"], 1)     # full-M prefill (token_difr)
    prx_s = ts["proxy"] / max(tc["proxy"], 1)             # cheap proxy prefill (SpecVerifier)
    seq_len = backend.max_prompt_tokens + N_TOKENS
    flops_ref = 2 * backend.n_params                      # ~forward FLOPs per token
    flops_prx = 2 * backend.proxy_n_params

    def per_1m_usd(sec_per_seq):
        return (sec_per_seq / N_TOKENS) * 1e6 / 3600.0 * USD_PER_HR

    print(f"  {'verifier':>26} | {'ms / seq':>9} | {'GFLOP / tok':>11} | {'$ / 1M tok':>10}")
    print("  " + "-" * 68)
    print(f"  {'token_difr (recompute M)':>26} | {ref_s*1e3:>9.1f} | "
          f"{flops_ref/1e9:>11.1f} | {per_1m_usd(ref_s):>10.3f}")
    print(f"  {'spec_accept (proxy q)':>26} | {prx_s*1e3:>9.1f} | "
          f"{flops_prx/1e9:>11.1f} | {per_1m_usd(prx_s):>10.3f}")
    print("  " + "-" * 68)
    print(f"  cost saving (recompute / proxy):  {ref_s/max(prx_s,1e-9):>5.1f}x wall-clock   "
          f"|   {ratio:>5.1f}x FLOPs")
    print(f"  (GPU priced at ${USD_PER_HR:.2f}/hr; seq_len~{seq_len} tok; "
          f"perf_counter GPU-synchronised)")

    try:
        make_figure(rows, ref_s, prx_s, ratio)
    except Exception as e:
        print(f"\n(figure skipped: {e})")
    print(f"\ntotal {time.time()-t0:.1f}s")


def make_figure(rows, ref_s, prx_s, ratio):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    body = [r for r in rows if r[0] != "(honest null)"]
    labels = [r[0] for r in body]
    spec = [r[1] for r in body]
    td = [r[3] for r in body]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(labels)); w = 0.38
    ax1.bar(x - w / 2, spec, w, label="spec_accept (proxy)", color="#4C78A8")
    ax1.bar(x + w / 2, td, w, label="token_difr (recompute M)", color="#E45756")
    ax1.axhline(0.5, ls=":", c="gray", lw=1)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("detection AUC"); ax1.set_ylim(0, 1.05)
    ax1.set_title("Performance: SpeculativeVerifier vs recompute")
    ax1.legend(fontsize=8, loc="lower left")

    names = ["spec_accept\n(proxy q)", "token_difr\n(recompute M)"]
    costs = [prx_s * 1e3, ref_s * 1e3]
    ax2.bar(names, costs, color=["#4C78A8", "#E45756"])
    ax2.set_ylabel("GPU ms / verified sequence")
    ax2.set_title(f"Cost: {ref_s/max(prx_s,1e-9):.1f}x cheaper ({ratio:.1f}x FLOPs)")
    for i, c in enumerate(costs):
        ax2.text(i, c, f"{c:.1f} ms", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = Path(FIG); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
