"""Information-directed verification on a real model: rank by I, aggregate by matched filter.

`docs/TRIAGE_AND_AUDIT_COST.md` Part 1 is a clean negative: a learned head trained to
predict per-token **sensitivity** `Delta(t)` does not beat the hand-crafted `entropy`
signal it was ported to replace, and it closes by naming the suspect -- "per-token
surrogate sensitivity is not the quantity that maximizes batch-level detection".

`ivgym/infogain.py` derives what the quantity actually is. For the weighted, centered
batch statistic the detection power of an audited set is

    delta^2(A) = (b/n) * sum_{t in A} I(t),      I(t) = Delta(t)^2 / v(t)

with `v(t)` the position's HONEST variance (the nondeterminism), maximized by the
matched filter `w = Delta/v`. So sensitivity is the numerator only: a near-tie
position is sensitive *and* noisy, and ranking on `Delta` buys both. This experiment
measures whether fixing that helps, on the same model, attack, pool and protocol the
negative result was measured on -- so the only thing that changes is the target.

Two arms, crossed, each with an oracle ceiling
----------------------------------------------
  * **allocation** (which tokens to audit): `uniform` / `entropy` / `tie_margin` /
    `surprisal` / `sensitivity` / `info`. The last two are fit on the SAME probe
    data with the SAME nine Tier-0 features, differing only in target --
    `triage.ConfidenceHead` on `Delta` (a reproduction of the earlier head, refit
    here so the comparison is controlled) against `infogain.InfoModel`'s
    `I = Delta^2/v`.
  * **aggregation** (how to combine the audited evidence): the plain batch `mean`
    every number in this repo was measured with, against the `matched` filter
    (`TokenScores.weights/baseline`, zero off the audit mask).
  * **the oracle arms**: `oracle_info` (allocation) and `oracle_mf` (aggregation)
    read `Delta` and `v` off labeled honest/attack score pairs
    (`infogain.realized_moments`) instead of estimating them from the cheap proxy.
    They are not deployable; they are here because a negative without them is not
    diagnosable. The full 7 x 3 cross separates three failure modes: a deployable
    arm that loses while the oracle arm wins indicts the **Tier-0 estimator**; an
    oracle arm that also loses indicts the **derivation**; and the mixed cells
    (deployable ranking x oracle weights, and back) say which of the two fitted
    pieces -- `Delta` or `v` -- the estimator is getting wrong.

Both fits read `ref_logits` only during the one-time offline probe on a disjoint
honest prompt range -- the amortization `ProxyReference.fit` and
`triage.surrogate_sensitivity` already assume. At verification time everything
except the two oracle arms is Tier-0 (cheap proxy only).

The theory arm
--------------
`signal.pauc_of_capture` predicts the whole detection-vs-budget curve from the
information the *ranking* captures, which is computable before any recompute is
spent. Panel C of the figure plots that prediction against the measured curve: a
falsifiable statement about the Gaussian/independent-token model in `ivgym/signal.py`,
not a fit. It is predicted twice -- from the Tier-0 `I_hat` (deployable, but wrong if
either the model or the estimator is) and from the oracle `I` (which tests the model
alone).

    IVGYM_M=Qwen/Qwen3-1.7B IVGYM_PROXY=Qwen/Qwen3-0.6B \
        python -m experiments.exp_info_directed_gpu
Env: IVGYM_M, IVGYM_PROXY, IVGYM_TRAIN(16), IVGYM_PROMPTS(64), IVGYM_TOKENS(128),
     IVGYM_BATCH(400), IVGYM_PROBE_SIGMA(0.15), IVGYM_NPROBE(8), IVGYM_SEEDS(9),
     IVGYM_ORACLE_BINS(40), IVGYM_ATTACKS(quant_2bit,quant_4bit).

Writes `docs/results/info_directed.json` (+ `info_directed_params.json`, the fitted
model) and, via `experiments/plot_info_directed.py`,
`docs/figures/fig_info_directed.png`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, harness, infogain, signal, triage, verifiers
from ivgym.backends.hf_gpu import HFGPUBackend
from ivgym.core import SamplingSpec, VContext
from ivgym.sampling import gumbel_noise, position_seed

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
RES_DIR = ROOT / "docs" / "results"

M = os.environ.get("IVGYM_M", "Qwen/Qwen3-1.7B")
PROXY = os.environ.get("IVGYM_PROXY", "Qwen/Qwen3-0.6B")
N_TRAIN = int(os.environ.get("IVGYM_TRAIN", 16))
# Defaults ARE the headline configuration of docs/TRIAGE_AND_AUDIT_COST.md, so this
# experiment's numbers are directly comparable to Part 1's table: 64 x 128 tokens at
# batch 400 is 9.8% of the honest null split, under the ~10% batch/pool ratio ceiling
# that § 3 of that document shows inflated this repo's older AUCs.
N_EVAL = int(os.environ.get("IVGYM_PROMPTS", 64))
T = int(os.environ.get("IVGYM_TOKENS", 128))
BATCH = int(os.environ.get("IVGYM_BATCH", 400))
PROBE_SIGMA = float(os.environ.get("IVGYM_PROBE_SIGMA", 0.15))
N_PROBE = int(os.environ.get("IVGYM_NPROBE", 8))
# 9, not the 5 of docs/TRIAGE_AND_AUDIT_COST.md: seeds are pure CPU (the model is
# never touched again) and the paired-by-seed comparisons below are what this
# experiment turns on, so they are the cheapest resolution available here.
N_SEED = int(os.environ.get("IVGYM_SEEDS", 9))
EVAL_ATTACKS = [a for a in os.environ.get(
    "IVGYM_ATTACKS", "quant_2bit,quant_4bit").split(",") if a]

BUDGETS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
# `sensitivity` and `info` are the two deployable arms of the comparison; the four
# before them are the library's existing hand-crafted signals, unchanged.
# `oracle_info` and `oracle_mf` are NOT deployable -- they read the attack's own
# scores (`infogain.realized_moments`) -- and they are here to make a negative
# diagnosable: they are the same statistic with the estimation error removed, so a
# result that fails deployably and holds at the oracle indicts the Tier-0 estimator,
# while one that fails at the oracle too indicts the derivation itself.
VALUE_FNS = ["uniform", "entropy", "tie_margin", "surprisal", "sensitivity", "info",
             "oracle_info"]
AGGS = ["mean", "matched", "oracle_mf"]
ORACLE_BINS = int(os.environ.get("IVGYM_ORACLE_BINS", 40))


# --------------------------------------------------------------------------- io
def gen(backend, attack, spec, ids):
    """One sequence per prompt id (explicit ids so fit/eval ranges stay disjoint)."""
    return [backend.generate(pid, T, spec, attack) for pid in ids]


def _ctx(backend, seq, spec):
    return VContext(seq.prompt_id, [st.claimed_token for st in seq.steps], spec,
                    proxy_logits=np.stack([backend.proxy_logits(seq.prompt_id, st.position)
                                           for st in seq.steps]))


def features_of(backend, seqs, spec):
    """Flat `[N*T, F]` Tier-0 feature matrix in (sequence, step) order."""
    return np.concatenate([triage.feature_matrix(_ctx(backend, s, spec)) for s in seqs])


def probe_of(backend, seqs, spec, verifier, rng, benign_sigma):
    """Flat probe moments over a fit split. Reads `ref_logits` -- offline only."""
    parts = []
    for seq in seqs:
        toks = [st.claimed_token for st in seq.steps]
        ref = np.stack([backend.reference_logits(seq.prompt_id, st.position)
                        for st in seq.steps])
        gum = np.stack([gumbel_noise(backend.vocab,
                                     position_seed(spec.seed, seq.prompt_id, st.position))
                        for st in seq.steps])
        parts.append(infogain.probe_moments(ref, gum, toks, spec, verifier, rng,
                                            probe_sigma=PROBE_SIGMA,
                                            benign_sigma=benign_sigma, n_probe=N_PROBE))
    return {k: np.concatenate([p[k] for p in parts]) for k in ("m", "v", "delta", "info")}


def value_signals(x, head, model):
    """Every DEPLOYABLE allocation signal for one config, off ONE Tier-0 feature
    matrix. The `oracle_info` arm is added per attack inside the loop, since it is
    a function of that attack's own scores."""
    col = {n: x[:, triage.FEATURE_NAMES.index(n)] for n in
           ("entropy", "tie_margin", "surprisal", "rel_position")}
    return {"uniform": np.ones(len(x)), "entropy": col["entropy"],
            "tie_margin": col["tie_margin"], "surprisal": col["surprisal"],
            "sensitivity": head.score(x, col["rel_position"]),
            "info": model.info(x)}


def cells(full, td, vals, mf, oracle_mf):
    """Every `(signal, budget, aggregation)` `TokenScores` for one config.

    A Tier-1 token score does not depend on which OTHER tokens were audited, so
    every cell is the full-budget array masked -- bit-identical to re-running the
    driver (`tests/test_triage_and_cost.py`). `mf` / `oracle_mf` are
    `(weights, baseline)` pairs for the two weighted aggregations.
    """
    wb = {"mean": (None, None), "matched": mf, "oracle_mf": oracle_mf}
    return {(vf, b, ag): harness.rescore_at_budget(
                full, [td], vals[vf], b, weights=wb[ag][0], baseline=wb[ag][1])
            for vf in VALUE_FNS for b in BUDGETS for ag in AGGS}


def auc_seeds(h_sc, a_sc, defs):
    """AUC@FPR<=0.5% per protocol seed -- the seed drives the honest calib/eval split
    and the batch resampling, i.e. every Monte-Carlo source inside `evaluate`.

    The per-seed values are kept (not just their mean and sd) because seed `s` means
    the same protocol draw in EVERY cell, so two cells can be differenced seed by
    seed. That paired difference is far tighter than the unpaired sds allow -- it is
    what lets this experiment separate two triage signals, which Part 1 of
    `docs/TRIAGE_AND_AUDIT_COST.md` explicitly could not.
    """
    return [float(harness.evaluate(h_sc, a_sc, defs, [BATCH], seed=7 + s)[0].auc)
            for s in range(N_SEED)]


def paired(a: list[float], b: list[float]) -> dict:
    """Paired difference `a - b` across shared protocol seeds: mean, sem, and the
    ratio (a t statistic on `len(a)-1` df). `sem = 0` with a nonzero mean means every
    seed moved the same way, which is a stronger statement than a large t, so the
    ratio is reported as `inf` rather than hidden."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    sem = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    t = float(d.mean() / sem) if sem > 0 else (np.inf if d.mean() != 0 else 0.0)
    return {"diff": float(d.mean()), "sem": sem, "t": t, "n": int(n)}


def transformed_d_prime(h_raw, a_raw, w_h, m_h, w_a, m_a, winsor_pct=99.9):
    """Per-token d' of the MATCHED-FILTER statistic, on `evaluate`'s scale.

    `evaluate` winsorizes the raw score at an honest percentile and only then applies
    `weights * (score - baseline)`, so the theory prediction has to be computed the
    same way round or it is predicting a different statistic."""
    cap = np.percentile(h_raw[np.isfinite(h_raw)], winsor_pct)
    h = w_h * (np.minimum(h_raw, cap) - m_h)
    a = w_a * (np.minimum(a_raw, cap) - m_a)
    return signal.per_token_stats(h, a, winsor_pct=None)["d_prime"]


# ------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"Information-directed verification   M={M}  proxy={PROXY}")
    print(f"  allocation: {', '.join(VALUE_FNS)}")
    print(f"  aggregation: {', '.join(AGGS)}   budgets: "
          f"{', '.join(f'{b:.0%}' for b in BUDGETS)}")
    print("=" * 78, flush=True)

    backend = HFGPUBackend(model_name=M, proxy_model_name=PROXY)
    spec = SamplingSpec()
    td = verifiers.get("token_difr")
    honest_attack = attacks.get("honest")
    rng = np.random.default_rng(0)
    # The provider and the verifier each add independent benign noise, so the
    # nondeterminism a re-scored honest row faces has scale sqrt(2)*verifier_sigma.
    benign = float(np.sqrt(2.0) * backend.verifier_sigma)
    print(f"loaded  M={backend.n_params/1e9:.2f}B  proxy={backend.proxy_n_params/1e9:.2f}B "
          f"benign_sigma={benign:.4f} [{time.time()-t0:.0f}s]", flush=True)

    # ---- 1. FIT on a disjoint honest prompt range -------------------------
    fit_ids = list(range(N_TRAIN))
    eval_ids = list(range(N_TRAIN, N_TRAIN + N_EVAL))
    assert N_TRAIN + N_EVAL <= len(backend.prompts), (
        f"prompt bank holds {len(backend.prompts)}; need {N_TRAIN + N_EVAL}")

    fit_seqs = gen(backend, honest_attack, spec, fit_ids)
    x_fit = features_of(backend, fit_seqs, spec)
    mom = probe_of(backend, fit_seqs, spec, td, rng, benign)
    h_fit = harness.verify(backend, fit_seqs, spec, [td]).scores["token_difr"]
    print(f"probed {len(x_fit)} honest tokens on prompts {fit_ids[0]}..{fit_ids[-1]} "
          f"({N_PROBE} probes/side, probe_sigma={PROBE_SIGMA}) [{time.time()-t0:.0f}s]",
          flush=True)
    print(f"  probe moments: mean m={mom['m'].mean():.4f}  mean v={mom['v'].mean():.4f} "
          f"(zero at {100*np.mean(mom['v'] == 0):.0f}% of positions)  "
          f"mean delta={mom['delta'].mean():.4f}")

    model = infogain.InfoModel().fit(x_fit, mom)
    # The sensitivity arm: the SAME features and the SAME probe, target `Delta` --
    # a controlled reproduction of the head in docs/TRIAGE_AND_AUDIT_COST.md Part 1.
    head = triage.ConfidenceHead().fit(x_fit, mom["delta"])
    fr = model.fit_report
    print(f"  InfoModel  R2(delta)={fr['r2_delta']:.3f}  R2(log v)={fr['r2_log_v']:.3f}  "
          f"R2(m)={fr['r2_m']:.3f}  spearman(I_hat, I_probe)={fr['spearman_info']:.3f}")
    print(f"  coef delta: " + "  ".join(f"{k} {v:+.2f}" for k, v in
                                        sorted(fr["coef_delta"].items(),
                                               key=lambda kv: -abs(kv[1]))[:5]))
    print(f"  coef log v: " + "  ".join(f"{k} {v:+.2f}" for k, v in
                                        sorted(fr["coef_log_v"].items(),
                                               key=lambda kv: -abs(kv[1]))[:5]),
          flush=True)

    verifiers.register_value_fn("info", infogain.info_value_fn(model))
    verifiers.register_value_fn("sensitivity", triage.head_value_fn(head))

    # ---- 2. HONEST eval pool ---------------------------------------------
    # Backend caches are per-prompt-id and overwritten by the next generate, so
    # every read for a config happens before the next config is generated.
    honest = gen(backend, honest_attack, spec, eval_ids)
    x_h = features_of(backend, honest, spec)
    h_vals = value_signals(x_h, head, model)
    w_h, m_h = infogain.matched_filter(model, x_h)
    h_full = harness.verify(backend, honest, spec, [td])
    h_raw = h_full.scores["token_difr"]
    n_tok = len(h_raw)
    print(f"honest eval pool: {n_tok} tokens, batch {BATCH} = "
          f"{BATCH/(0.5*n_tok):.1%} of the null split [{time.time()-t0:.0f}s]", flush=True)

    # Tier-0 information capture per (deployable signal, budget): computable with no
    # recompute at all, and the input to the theory prediction.
    i_hat = h_vals["info"]
    capture = {vf: [signal.info_capture(i_hat, harness.select_triaged(h_vals[vf], b))
                    for b in BUDGETS] for vf in VALUE_FNS if vf != "oracle_info"}

    # ---- 3. ATTACKS -------------------------------------------------------
    payload = {
        "M": M, "proxy": PROXY, "n_train": N_TRAIN, "n_eval": N_EVAL, "tokens": T,
        "batch": BATCH, "n_seed": N_SEED, "probe_sigma": PROBE_SIGMA,
        "n_probe": N_PROBE, "benign_sigma": benign, "budgets": BUDGETS,
        "value_fns": VALUE_FNS, "aggs": AGGS, "eval_attacks": EVAL_ATTACKS,
        "headline_attack": EVAL_ATTACKS[0], "eval_tokens": int(n_tok),
        "batch_frac_of_null": float(BATCH / (0.5 * n_tok)),
        "fit_report": fr, "capture": capture, "oracle_bins": ORACLE_BINS,
        "probe_summary": {k: float(np.mean(mom[k])) for k in ("m", "v", "delta")},
        "auc": {}, "auc_sd": {}, "theory": {}, "diagnostics": {},
        "capture_oracle": {}, "auc_seeds": {}, "comparisons": {},
    }

    for atk_name in EVAL_ATTACKS:
        a_seqs = gen(backend, attacks.get(atk_name), spec, eval_ids)
        x_a = features_of(backend, a_seqs, spec)
        a_vals = value_signals(x_a, head, model)
        w_a, m_a = infogain.matched_filter(model, x_a)
        a_full = harness.verify(backend, a_seqs, spec, [td])
        a_raw = a_full.scores["token_difr"]

        # The oracle arms: the SAME InfoModel over the SAME nine features, with its
        # targets read off this attack's labeled pairs instead of the honest-only
        # probe (`infogain.oracle_model`). Fit on the eval pool -- nine coefficients
        # over `n_tok` tokens -- so it bounds what perfect estimation of `(Delta, v)`
        # from these features would buy, and both sides are scored from their OWN
        # features exactly as the deployable arm is. Because the signal depends on
        # the attack, the honest cells are rebuilt per attack too.
        omodel = infogain.oracle_model(x_h, h_raw, a_raw)
        oracle_i = omodel.info(x_h)
        w_o, m_o = infogain.matched_filter(omodel, x_h)
        w_oa, m_oa = infogain.matched_filter(omodel, x_a)
        h_vals["oracle_info"] = oracle_i
        a_vals["oracle_info"] = omodel.info(x_a)
        h_sel = cells(h_full, td, h_vals, (w_h, m_h), (w_o, m_o))
        a_sel = cells(a_full, td, a_vals, (w_a, m_a), (w_oa, m_oa))

        rows, sds, per_seed = {}, {}, {}
        for ag in AGGS:
            for vf in VALUE_FNS:
                v = [auc_seeds(h_sel[(vf, b, ag)], a_sel[(vf, b, ag)], [td])
                     for b in BUDGETS]
                per_seed[f"{ag}/{vf}"] = v
                rows[f"{ag}/{vf}"] = [float(np.mean(s)) for s in v]
                sds[f"{ag}/{vf}"] = [float(np.std(s)) for s in v]
        payload["auc"][atk_name] = rows
        payload["auc_sd"][atk_name] = sds
        payload["auc_seeds"][atk_name] = per_seed

        # --- the comparisons this experiment exists to make, paired by seed ----
        # Each is a claim in `ivgym/infogain.py`'s docstring, one entry per budget.
        cmp_pairs = {
            "P1a_info_vs_sensitivity": ("matched/info", "matched/sensitivity"),
            "P1a_info_vs_sensitivity_mean_agg": ("mean/info", "mean/sensitivity"),
            "P1a_info_vs_best_handcrafted": ("matched/info", None),
            "P1b_matched_vs_mean_at_info": ("matched/info", "mean/info"),
            "P1b_matched_vs_mean_at_uniform": ("matched/uniform", "mean/uniform"),
            "oracle_alloc_vs_info": ("mean/oracle_info", "mean/info"),
            "oracle_agg_vs_mean": ("oracle_mf/uniform", "mean/uniform"),
            "oracle_both_vs_deployable": ("oracle_mf/oracle_info", "matched/info"),
            "oracle_both_vs_full_mean": ("oracle_mf/oracle_info", "mean/uniform"),
        }
        hand = [f"matched/{v}" for v in ("entropy", "tie_margin", "surprisal")]
        comparisons = {}
        for name, (lhs, rhs) in cmp_pairs.items():
            out = []
            for i, b in enumerate(BUDGETS):
                if rhs is None:      # vs whichever hand-crafted signal wins here
                    rkey = max(hand, key=lambda k: rows[k][i])
                    r = paired(per_seed[lhs][i], per_seed[rkey][i])
                    r["vs"] = rkey
                else:
                    r = paired(per_seed[lhs][i], per_seed[rhs][i])
                r["budget"] = b
                out.append(r)
            comparisons[name] = out
        payload["comparisons"][atk_name] = comparisons

        # --- the theory arm: predict the curve from the capture --------------
        # Deployably from the Tier-0 `I_hat`, and again from the ORACLE `I`. The
        # second prediction is the one that tests the Gaussian/independent-token
        # model of `ivgym/signal.py` on its own, with no estimation error in it.
        cap_o = {vf: [signal.info_capture(oracle_i, harness.select_triaged(h_vals[vf], b))
                      for b in BUDGETS] for vf in VALUE_FNS}
        payload["capture_oracle"][atk_name] = cap_o
        d_mean = signal.per_token_stats(h_raw, a_raw)["d_prime"]
        d_mf = transformed_d_prime(h_raw, a_raw, w_h, m_h, w_a, m_a)
        d_omf = transformed_d_prime(h_raw, a_raw, w_o, m_o, w_oa, m_oa)
        pred = [signal.pauc_of_capture(c, d_mf, BATCH) for c in capture["info"]]
        meas = rows["matched/info"]
        pred_o = [signal.pauc_of_capture(c, d_omf, BATCH) for c in cap_o["oracle_info"]]
        meas_o = rows["oracle_mf/oracle_info"]
        payload["theory"][atk_name] = {
            "d_prime_mean": d_mean, "d_prime_matched": d_mf,
            "d_prime_oracle_matched": d_omf,
            "predicted": pred, "measured": meas,
            "mae": float(np.mean(np.abs(np.array(pred) - np.array(meas)))),
            "oracle_predicted": pred_o, "oracle_measured": meas_o,
            "oracle_mae": float(np.mean(np.abs(np.array(pred_o) - np.array(meas_o)))),
            "predicted_full_batch_for_090": signal.batch_for_pauc(d_mf),
        }

        # --- diagnostics: how good is the Tier-0 estimate of I? ------------
        # Against the feature-level oracle (the ceiling arm) and, separately, the
        # position-level binned one -- which is biased low in `v` and so is read only
        # as "where does the information sit", never as a bound.
        binned_i = infogain.realized_info(h_raw, a_raw, ORACLE_BINS)
        payload["diagnostics"][atk_name] = {
            "spearman_info_vs_oracle": infogain._spearman(i_hat, oracle_i),
            "spearman_sensitivity_vs_oracle": infogain._spearman(
                h_vals["sensitivity"], oracle_i),
            "spearman_entropy_vs_oracle": infogain._spearman(h_vals["entropy"], oracle_i),
            "spearman_tie_vs_oracle": infogain._spearman(h_vals["tie_margin"], oracle_i),
            "spearman_delta_hat_vs_oracle_delta": infogain._spearman(
                model.sensitivity(x_h), omodel.sensitivity(x_h)),
            "spearman_v_hat_vs_oracle_v": infogain._spearman(
                model.noise(x_h), omodel.noise(x_h)),
            "spearman_info_vs_binned": infogain._spearman(i_hat, binned_i),
            "spearman_oracle_vs_binned": infogain._spearman(oracle_i, binned_i),
            "oracle_fit_report": omodel.fit_report,
        }

        print(f"\n  {atk_name}:  per-token d' mean-agg {d_mean:.4f} -> "
              f"matched-filter {d_mf:.4f} -> oracle matched {d_omf:.4f}")
        # One table per aggregation: 3 x 7 cells on one row is unreadable, and the
        # comparison that matters is down a column (allocation) or across tables
        # (aggregation), never diagonally.
        for ag in AGGS:
            print(f"  [{ag}]{'budget':>9}" +
                  "".join(f"{vf[:11]:>13}" for vf in VALUE_FNS))
            for i, b in enumerate(BUDGETS):
                print(f"  {'':>6} {b:>8.0%}" +
                      "".join(f"{rows[f'{ag}/{vf}'][i]:>8.3f}"
                              f"{'':>1}({sds[f'{ag}/{vf}'][i]:.3f})"
                              for vf in VALUE_FNS))
        print(f"  theory (matched/info):        predicted " +
              " ".join(f"{p:.3f}" for p in pred))
        print(f"                                measured  " +
              " ".join(f"{m:.3f}" for m in meas) +
              f"   MAE {payload['theory'][atk_name]['mae']:.3f}")
        print(f"  theory (oracle_mf/oracle_info): predicted " +
              " ".join(f"{p:.3f}" for p in pred_o))
        print(f"                                  measured  " +
              " ".join(f"{m:.3f}" for m in meas_o) +
              f"   MAE {payload['theory'][atk_name]['oracle_mae']:.3f}")
        dg = payload["diagnostics"][atk_name]
        print(f"  spearman vs oracle I: info {dg['spearman_info_vs_oracle']:+.3f}  "
              f"sensitivity {dg['spearman_sensitivity_vs_oracle']:+.3f}  "
              f"entropy {dg['spearman_entropy_vs_oracle']:+.3f}  "
              f"tie_margin {dg['spearman_tie_vs_oracle']:+.3f}")
        print(f"  spearman of the two fitted pieces: delta_hat vs oracle delta "
              f"{dg['spearman_delta_hat_vs_oracle_delta']:+.3f}  "
              f"v_hat vs oracle v {dg['spearman_v_hat_vs_oracle_v']:+.3f}")
        # The claims, paired by seed. A verdict per comparison at the budget where
        # the effect is largest -- printed with its t so a null reads as a null and
        # not as "no difference found".
        print(f"  {'paired comparison (best budget)':<38}{'diff':>8}{'sem':>8}"
              f"{'t':>7}   verdict")
        for name, out in comparisons.items():
            k = int(np.argmax([abs(r["diff"]) for r in out]))
            r = out[k]
            verdict = ("supports" if r["diff"] > 0 and abs(r["t"]) >= 2 else
                       "REFUTES" if r["diff"] < 0 and abs(r["t"]) >= 2 else "null")
            print(f"  {name + ' @ %.0f%%' % (100 * r['budget']):<38}"
                  f"{r['diff']:>+8.3f}{r['sem']:>8.3f}{r['t']:>7.2f}   {verdict}")
        print(f"  [{time.time()-t0:.0f}s]", flush=True)

    out = RES_DIR / "info_directed.json"
    out.write_text(json.dumps(payload, indent=2))
    (RES_DIR / "info_directed_params.json").write_text(json.dumps({
        "M": M, "proxy": PROXY, "tokens": T, "probe_sigma": PROBE_SIGMA,
        "n_probe": N_PROBE, "benign_sigma": benign, "fit_prompt_ids": fit_ids,
        "eval_prompt_ids": eval_ids, "info_model": model.to_dict(),
        "sensitivity_head": head.to_dict()}, indent=2))
    print(f"\nwrote {out}")
    try:
        from experiments import plot_info_directed
        plot_info_directed.main([])
    except Exception as e:                      # a figure failure must not lose data
        print(f"[figure skipped: {type(e).__name__}: {e}]")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
