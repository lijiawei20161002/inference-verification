"""A learned, calibrated triage head vs the four hand-crafted value signals.

Ports the *confidence head* of DSpark (arXiv:2607.05147) into the verification
game. DSpark predicts per-position acceptance probability from cheap draft-side
features, trains it with BCE against analytic acceptance rates, and post-hoc
calibrates it so a scheduler can spend against a cost model. Here the same
machinery predicts, from proxy-only features, *which positions an expensive
recompute would actually learn something from* -- and the resulting value signal
is dropped straight into `harness.verify(budget<1)` alongside `uniform` /
`entropy` / `tie_margin` / `surprisal`.

Reported (all under the standardized `EvalConfig` protocol, so numbers are
comparable to every other experiment in the repo):

  1. Triage Pareto: detection AUC@FPR<=0.5% vs recompute budget, per value signal.
  2. Calibration: reliability of the head before/after Sequential Temperature
     Scaling, with ECE.
  3. What the head learned: coefficient per proxy feature.
  4. Cross-attack transfer: the deployable honest-only head, and the oracle head
     trained on one attack and triaged against another.

Train/eval hygiene: FOUR disjoint honest prompt ranges, because each answers a
different question and sharing any two of them makes one of the answers in-sample.

  1. `train`    -- fit the head (BCE on surrogate-sensitivity labels)
  2. `sts_fit`  -- fit Sequential Temperature Scaling (NLL)
  3. `sts_eval` -- SCORE the reliability curve / ECE. Held out from (2), so
                   "did calibration help" is an out-of-sample question. Sharing it
                   with (2) measures how well STS fit its own training split,
                   which it can only ever answer yes to.
  4. `eval`     -- the detection AUC sweep.

The deployable head never sees an attack, and reads `ref_logits` only during the
one-time offline fit -- the same trusted-M-run amortization `ProxyReference.fit`
assumes.

The fitted head is written to `docs/results/confidence_head_params.json` so
`exp_prefix_cost_gpu.py` can schedule the prefix scheduler against this exact
calibrated head instead of refitting one (or guessing with `tie_margin`).

    IVGYM_M=Qwen/Qwen3-1.7B IVGYM_PROXY=Qwen/Qwen3-0.6B \
        python -m experiments.exp_confidence_head_gpu
Env: IVGYM_M, IVGYM_PROXY, IVGYM_TRAIN(16), IVGYM_PROMPTS(32), IVGYM_TOKENS(96),
     IVGYM_BATCH(200), IVGYM_PROBE_SIGMA(0.15), IVGYM_NPROBE(8),
     IVGYM_ATTACKS(quant_4bit,kv_fp8,bug_k2), IVGYM_SEEDS(1).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, harness, triage, verifiers
from ivgym.backends.hf_gpu import HFGPUBackend
from ivgym.core import SamplingSpec, VContext
from experiments import plot_triage
from ivgym.sampling import gumbel_noise, position_seed

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
RES_DIR = ROOT / "docs" / "results"

M = os.environ.get("IVGYM_M", "Qwen/Qwen3-1.7B")
PROXY = os.environ.get("IVGYM_PROXY", "Qwen/Qwen3-0.6B")
N_TRAIN = int(os.environ.get("IVGYM_TRAIN", 16))
# Defaults ARE the headline configuration of docs/TRIAGE_AND_AUDIT_COST.md: batch
# 400 is 9.8% of the 8192-token honest split, under the ~10% ratio ceiling that
# § 3 of that doc shows is what inflated this repo's older AUCs. The old 32 x 96 /
# batch-200 default sat at 13% and produced an ordering that did not replicate.
N_EVAL = int(os.environ.get("IVGYM_PROMPTS", 64))
T = int(os.environ.get("IVGYM_TOKENS", 128))
BATCH = int(os.environ.get("IVGYM_BATCH", 400))
PROBE_SIGMA = float(os.environ.get("IVGYM_PROBE_SIGMA", 0.15))
N_PROBE = int(os.environ.get("IVGYM_NPROBE", 8))

# Single-seed runs cannot see the sd this table is read against; 5 is the minimum
# that makes a cell comparable to its neighbour.
N_SEED = int(os.environ.get("IVGYM_SEEDS", 5))

BUDGETS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
BASELINES = ["uniform", "entropy", "tie_margin", "surprisal"]
# First entry is the headline: quant_2bit is the rung with detection headroom at
# this batch/pool ratio, quant_4bit is kept as the weaker rung so the table shows
# what happens when the ceiling drops (full recompute 0.867 vs 0.629).
EVAL_ATTACKS = [a for a in os.environ.get(
    "IVGYM_ATTACKS", "quant_2bit,quant_4bit").split(",") if a]
# The oracle head gets labeled attack data from THIS deviation only; the
# cross-attack columns then show whether that label transfers.
ORACLE_ATTACK = os.environ.get("IVGYM_ORACLE_ATTACK", "quant_2bit")
TRANSFER_BUDGET = 0.20

# --------------------------------------------------------------------------- io
def gen(backend, attack, spec, ids):
    """Generate one sequence per prompt id (explicit ids so train/eval prompt
    ranges can be disjoint -- `harness.generate_dataset` always uses 0..N-1)."""
    return [backend.generate(pid, T, spec, attack) for pid in ids]


def _proxy_rows(backend, seq):
    return np.stack([backend.proxy_logits(seq.prompt_id, st.position) for st in seq.steps])


def _ctx(backend, seq, spec):
    return VContext(seq.prompt_id, [st.claimed_token for st in seq.steps], spec,
                    proxy_logits=_proxy_rows(backend, seq))


def features_of(backend, seqs, spec):
    """Flat `[N*T, F]` proxy-only feature matrix in (sequence, step) order."""
    return np.concatenate([triage.feature_matrix(_ctx(backend, s, spec)) for s in seqs])


def value_signals(x, heads):
    """Every triage value signal for one config, off ONE feature matrix.

    `token_values` would re-stack the `[T, V]` proxy rows and re-softmax them once
    per value function. Three of the four hand-crafted signals ARE columns of
    `triage.feature_matrix` by construction (asserted in
    `tests/test_triage_and_cost.py::test_features_are_tier0_and_well_formed`), and
    both heads read the same matrix -- so one pass serves all six. `uniform` is
    constant by definition; `select_triaged` breaks its ties randomly, which is
    what makes it the equal-cost random-subsample control.
    """
    col = {n: x[:, triage.FEATURE_NAMES.index(n)] for n in
           ("entropy", "tie_margin", "surprisal", "rel_position")}
    out = {"uniform": np.ones(len(x)), "entropy": col["entropy"],
           "tie_margin": col["tie_margin"], "surprisal": col["surprisal"]}
    for name, h in heads.items():
        out[name] = h.score(x, col["rel_position"])
    return out


def auc_ms(h_sc, a_sc, defs, batch=None, n_seed=None):
    """AUC@FPR<=0.5% as (mean, sd) over `n_seed` independent protocol seeds.

    The seed drives the honest calib/eval split and the batch resampling -- every
    Monte-Carlo source inside `evaluate`. Reporting the sd is what makes the
    comparison between two triage signals falsifiable: the 2026-07-29 run's
    "learned 0.566 vs entropy 0.573" is only a finding if the sd is well under
    0.007, and it is not.
    """
    batch = BATCH if batch is None else batch
    n_seed = N_SEED if n_seed is None else n_seed
    v = [harness.evaluate(h_sc, a_sc, defs, [batch], seed=7 + s)[0].auc
         for s in range(n_seed)]
    return float(np.mean(v)), float(np.std(v))


def sensitivity_of(backend, seqs, spec, td, rng):
    """Flat surrogate-sensitivity labels. Reads `ref_logits` -- offline fit only."""
    out = []
    for seq in seqs:
        toks = [st.claimed_token for st in seq.steps]
        ref = np.stack([backend.reference_logits(seq.prompt_id, st.position)
                        for st in seq.steps])
        gum = np.stack([gumbel_noise(backend.vocab,
                                     position_seed(spec.seed, seq.prompt_id, st.position))
                        for st in seq.steps])
        out.append(triage.surrogate_sensitivity(ref, gum, toks, spec, td, rng,
                                                PROBE_SIGMA, N_PROBE))
    return np.concatenate(out)


# ------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"Learned triage head (DSpark confidence head, ported)  M={M}  proxy={PROXY}")
    print("=" * 78, flush=True)

    backend = HFGPUBackend(model_name=M, proxy_model_name=PROXY)
    spec = SamplingSpec()
    td = verifiers.get("token_difr")
    honest_attack = attacks.get("honest")
    rng = np.random.default_rng(0)
    print(f"loaded  M={backend.n_params/1e9:.2f}B  proxy={backend.proxy_n_params/1e9:.2f}B "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- 1. TRAIN the head on a disjoint honest prompt range --------------
    # Four disjoint honest ranges: fit the head | fit STS | SCORE calibration |
    # evaluate detection. `sts_eval` is what makes panel A out-of-sample.
    train_ids = list(range(N_TRAIN))
    cal_ids = list(range(N_TRAIN, 2 * N_TRAIN))
    sts_ids = list(range(2 * N_TRAIN, 3 * N_TRAIN))
    eval_ids = list(range(3 * N_TRAIN, 3 * N_TRAIN + N_EVAL))
    assert 3 * N_TRAIN + N_EVAL <= len(backend.prompts), (
        f"prompt bank holds {len(backend.prompts)}; need "
        f"{3*N_TRAIN + N_EVAL} for four disjoint honest splits")

    tr_seqs = gen(backend, honest_attack, spec, train_ids)
    x_tr = features_of(backend, tr_seqs, spec)
    s_tr = sensitivity_of(backend, tr_seqs, spec, td, rng)
    h_tr = harness.verify(backend, tr_seqs, spec, [td]).scores["token_difr"]
    head = triage.ConfidenceHead().fit(x_tr, s_tr)
    print(f"head fit on {len(x_tr)} honest tokens (disjoint prompts {train_ids[0]}..{train_ids[-1]}); "
          f"BCE {head.history[0]:.4f} -> {head.history[-1]:.4f} [{time.time()-t0:.0f}s]", flush=True)

    # ---- 1b. ORACLE head: same features, but labeled with REAL deviation ----
    # Trained on paired honest/attack Tier-1 scores over the SAME train prompts,
    # then evaluated held-out exactly like the deployable head. Isolates one
    # variable: surrogate probe labels vs real labeled attack data.
    ora_seqs = gen(backend, attacks.get(ORACLE_ATTACK), spec, train_ids)
    a_tr = harness.verify(backend, ora_seqs, spec, [td]).scores["token_difr"]
    oracle_head = triage.ConfidenceHead().fit(x_tr, triage.paired_effect_size(h_tr, a_tr))
    print(f"oracle head fit on the same features with {ORACLE_ATTACK} labels; "
          f"BCE {oracle_head.history[0]:.4f} -> {oracle_head.history[-1]:.4f} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- 2. CALIBRATE (STS) on a third split, SCORE it on a fourth --------
    # The 2026-07-29 run fit STS and scored its reliability on the same split, so
    # "did calibration help" was answered in-sample -- a question STS can only
    # ever answer yes to, since NLL is what it minimizes there. Both splits are
    # materialized BEFORE `calibrate` so the raw (uncalibrated) probability can be
    # read off the held-out split too.
    pos_of = lambda x: x[:, triage.FEATURE_NAMES.index("rel_position")]
    cal_seqs = gen(backend, honest_attack, spec, cal_ids)
    x_cal = features_of(backend, cal_seqs, spec)
    s_cal = sensitivity_of(backend, cal_seqs, spec, td, rng)
    sts_seqs = gen(backend, honest_attack, spec, sts_ids)
    x_sts = features_of(backend, sts_seqs, spec)
    s_sts = sensitivity_of(backend, sts_seqs, spec, td, rng)
    pos_cal, pos_sts = pos_of(x_cal), pos_of(x_sts)

    p_raw_in, p_raw_ho = head.score(x_cal, pos_cal), head.score(x_sts, pos_sts)
    head.calibrate(x_cal, s_cal, pos_cal)
    p_sts_in, p_sts_ho = head.score(x_cal, pos_cal), head.score(x_sts, pos_sts)

    def _rel(p, s):
        # The label threshold is taken on the split being SCORED, matching how
        # `fit`/`calibrate` binarize their own splits.
        y = (s > np.quantile(s, head.label_quantile)).astype(float)
        c, a, _, ece = triage.reliability(p, y)
        return {"conf": c.tolist(), "acc": a.tolist()}, ece

    r_raw_ho, ece_raw = _rel(p_raw_ho, s_sts)      # HELD OUT: the honest question
    r_sts_ho, ece_sts = _rel(p_sts_ho, s_sts)
    r_raw_in, ece_raw_in = _rel(p_raw_in, s_cal)   # in-sample, for the record
    r_sts_in, ece_sts_in = _rel(p_sts_in, s_cal)
    print(f"STS: temperature {head.calibrator.temperature():.3f}  "
          f"ECE held-out {ece_raw:.4f} -> {ece_sts:.4f}   "
          f"(in-sample {ece_raw_in:.4f} -> {ece_sts_in:.4f}) [{time.time()-t0:.0f}s]",
          flush=True)

    verifiers.register_value_fn("learned", triage.head_value_fn(head))
    verifiers.register_value_fn("oracle", triage.head_value_fn(oracle_head))

    # ---- 3. EVALUATE on the held-out prompt range -------------------------
    # Backend caches are per-prompt-id and overwritten by each generate, so all
    # reads for a config must happen before the next config is generated.
    heads = {"learned": head, "oracle": oracle_head}
    value_fns = BASELINES + list(heads)
    honest = gen(backend, honest_attack, spec, eval_ids)
    h_vals = value_signals(features_of(backend, honest, spec), heads)
    h_full = harness.verify(backend, honest, spec, [td])
    # A Tier-1 token score does not depend on which OTHER tokens were audited, so
    # every budget is the full-budget array masked -- bit-identical to re-running
    # the driver, and it makes a 6-signal x N-budget sweep affordable.
    h_sel = {(vf, b): harness.rescore_at_budget(h_full, [td], h_vals[vf], b)
             for vf in value_fns for b in BUDGETS}
    print(f"honest eval materialized ({len(h_full.scores['token_difr'])} tokens) "
          f"[{time.time()-t0:.0f}s]", flush=True)

    pareto, pareto_sd, transfer, transfer_sd = {}, {}, {}, {}
    b_t = TRANSFER_BUDGET
    for atk_name in EVAL_ATTACKS:
        a_seqs = gen(backend, attacks.get(atk_name), spec, eval_ids)
        a_vals = value_signals(features_of(backend, a_seqs, spec), heads)
        a_full = harness.verify(backend, a_seqs, spec, [td])
        a_sel = {(vf, b): harness.rescore_at_budget(a_full, [td], a_vals[vf], b)
                 for vf in value_fns for b in BUDGETS}
        ms = {vf: [auc_ms(h_sel[(vf, b)], a_sel[(vf, b)], [td]) for b in BUDGETS]
              for vf in value_fns}
        pareto[atk_name] = {vf: [m for m, _ in ms[vf]] for vf in value_fns}
        pareto_sd[atk_name] = {vf: [s for _, s in ms[vf]] for vf in value_fns}
        i_t = BUDGETS.index(b_t)
        transfer[atk_name] = {vf: pareto[atk_name][vf][i_t] for vf in value_fns}
        transfer_sd[atk_name] = {vf: pareto_sd[atk_name][vf][i_t] for vf in value_fns}
        f_m, f_s = auc_ms(h_full, a_full, [td])
        transfer[atk_name]["full"], transfer_sd[atk_name]["full"] = f_m, f_s
        print(f"  {atk_name}: @{b_t:.0%} " + "  ".join(
            f"{vf} {transfer[atk_name][vf]:.3f}+-{transfer_sd[atk_name][vf]:.3f}"
            for vf in ["entropy", "tie_margin", "learned", "oracle", "full"])
            + f" [{time.time()-t0:.0f}s]", flush=True)

    payload = {
        "M": M, "proxy": PROXY, "n_train": N_TRAIN, "n_eval": N_EVAL, "tokens": T,
        "batch": BATCH, "probe_sigma": PROBE_SIGMA, "n_probe": N_PROBE,
        "budgets": BUDGETS, "value_fns": value_fns, "eval_attacks": EVAL_ATTACKS,
        "headline_attack": EVAL_ATTACKS[0], "transfer_budget": TRANSFER_BUDGET,
        "oracle_attack": ORACLE_ATTACK, "n_seed": N_SEED,
        "eval_tokens": int(len(h_full.scores["token_difr"])),
        "batch_frac_of_null": float(BATCH / (0.5 * len(h_full.scores["token_difr"]))),
        "pareto": pareto, "pareto_sd": pareto_sd,
        "transfer": transfer, "transfer_sd": transfer_sd,
        "weights": head.weights(),
        "oracle_weights": oracle_head.weights(),
        # `reliability` is the HELD-OUT curve (fourth honest split) -- the figure's
        # panel A reads this key, so the panel is out-of-sample by construction.
        "reliability": {
            "raw": r_raw_ho, "sts": r_sts_ho,
            "ece_raw": ece_raw, "ece_sts": ece_sts,
            "temperature": head.calibrator.temperature(),
            "bias": head.calibrator.bias.tolist(),
            "held_out": True, "n_sts_fit_prompts": N_TRAIN,
            "n_sts_eval_prompts": N_TRAIN,
        },
        "reliability_insample": {
            "raw": r_raw_in, "sts": r_sts_in,
            "ece_raw": ece_raw_in, "ece_sts": ece_sts_in, "held_out": False,
        },
    }
    out = RES_DIR / "confidence_head.json"
    out.write_text(json.dumps(payload, indent=2))
    # The fitted+calibrated head itself, so the prefix scheduler can admit against
    # THIS head (exp_prefix_cost_gpu.py) without a second, differently-fit copy.
    params = RES_DIR / "confidence_head_params.json"
    params.write_text(json.dumps({
        "M": M, "proxy": PROXY, "n_train": N_TRAIN, "tokens": T,
        "probe_sigma": PROBE_SIGMA, "n_probe": N_PROBE,
        "train_prompt_ids": train_ids, "sts_fit_prompt_ids": cal_ids,
        "sts_eval_prompt_ids": sts_ids, "eval_prompt_ids": eval_ids,
        "learned": head.to_dict(), "oracle": oracle_head.to_dict(),
    }, indent=2))

    # ---- 4. report + figures ---------------------------------------------
    print(f"\n{'budget':>8}" + "".join(f"{v:>16}" for v in value_fns))
    print("  " + "-" * (6 + 16 * len(value_fns)))
    a0 = EVAL_ATTACKS[0]
    for i, b in enumerate(BUDGETS):
        print(f"{b*100:>7.0f}%" + "".join(
            f"{pareto[a0][v][i]:>11.3f}+-{pareto_sd[a0][v][i]:>4.3f}"
            for v in value_fns))
    print(f"\n  (AUC mean +- sd over {N_SEED} protocol seeds; a gap smaller than "
          f"~2 sd is not a finding)")
    print(f"\nwrote {out}\nwrote {params}")
    plot_triage.main([])
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
