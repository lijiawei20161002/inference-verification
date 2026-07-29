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

Train/eval hygiene: the head is fit on a DISJOINT honest prompt range from the
one it is evaluated on, and STS is fit on a third honest split. The deployable
head never sees an attack, and reads `ref_logits` only during the one-time
offline fit -- the same trusted-M-run amortization `ProxyReference.fit` assumes.

    IVGYM_M=Qwen/Qwen3-1.7B IVGYM_PROXY=Qwen/Qwen3-0.6B \
        python -m experiments.exp_confidence_head_gpu
Env: IVGYM_M, IVGYM_PROXY, IVGYM_TRAIN(16), IVGYM_PROMPTS(32), IVGYM_TOKENS(96),
     IVGYM_BATCH(200), IVGYM_PROBE_SIGMA(0.15), IVGYM_NPROBE(8).
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
N_EVAL = int(os.environ.get("IVGYM_PROMPTS", 32))
T = int(os.environ.get("IVGYM_TOKENS", 96))
BATCH = int(os.environ.get("IVGYM_BATCH", 200))
PROBE_SIGMA = float(os.environ.get("IVGYM_PROBE_SIGMA", 0.15))
N_PROBE = int(os.environ.get("IVGYM_NPROBE", 8))

BUDGETS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
BASELINES = ["uniform", "entropy", "tie_margin", "surprisal"]
EVAL_ATTACKS = ["quant_4bit", "kv_fp8", "bug_k2"]
# The oracle head gets labeled attack data from THIS deviation only; the
# cross-attack columns then show whether that label transfers.
ORACLE_ATTACK = os.environ.get("IVGYM_ORACLE_ATTACK", "quant_4bit")
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
    train_ids = list(range(N_TRAIN))
    cal_ids = list(range(N_TRAIN, 2 * N_TRAIN))
    eval_ids = list(range(2 * N_TRAIN, 2 * N_TRAIN + N_EVAL))
    assert 2 * N_TRAIN + N_EVAL <= len(backend.prompts), "prompt bank too small"

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

    # ---- 2. CALIBRATE (STS) on a third honest split ----------------------
    cal_seqs = gen(backend, honest_attack, spec, cal_ids)
    x_cal = features_of(backend, cal_seqs, spec)
    s_cal = sensitivity_of(backend, cal_seqs, spec, td, rng)
    pos_cal = x_cal[:, triage.FEATURE_NAMES.index("rel_position")]
    p_raw = head.score(x_cal, pos_cal)
    head.calibrate(x_cal, s_cal, pos_cal)
    p_sts = head.score(x_cal, pos_cal)
    y_cal = (s_cal > np.quantile(s_cal, head.label_quantile)).astype(float)
    c_raw, a_raw, _, ece_raw = triage.reliability(p_raw, y_cal)
    c_sts, a_sts, _, ece_sts = triage.reliability(p_sts, y_cal)
    print(f"STS: temperature {head.calibrator.temperature():.3f}  "
          f"ECE {ece_raw:.4f} -> {ece_sts:.4f} [{time.time()-t0:.0f}s]", flush=True)

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

    pareto, transfer = {}, {}
    b_t = TRANSFER_BUDGET
    for atk_name in EVAL_ATTACKS:
        a_seqs = gen(backend, attacks.get(atk_name), spec, eval_ids)
        a_vals = value_signals(features_of(backend, a_seqs, spec), heads)
        a_full = harness.verify(backend, a_seqs, spec, [td])
        a_sel = {(vf, b): harness.rescore_at_budget(a_full, [td], a_vals[vf], b)
                 for vf in value_fns for b in BUDGETS}
        pareto[atk_name] = {
            vf: [harness.evaluate(h_sel[(vf, b)], a_sel[(vf, b)], [td], [BATCH],
                                  seed=7)[0].auc for b in BUDGETS]
            for vf in value_fns}
        i_t = BUDGETS.index(b_t)
        transfer[atk_name] = {vf: pareto[atk_name][vf][i_t] for vf in value_fns}
        transfer[atk_name]["full"] = harness.evaluate(h_full, a_full, [td], [BATCH],
                                                      seed=7)[0].auc
        print(f"  {atk_name}: @{b_t:.0%} " + "  ".join(
            f"{vf} {transfer[atk_name][vf]:.3f}" for vf in
            ["entropy", "tie_margin", "learned", "oracle", "full"])
            + f" [{time.time()-t0:.0f}s]", flush=True)

    payload = {
        "M": M, "proxy": PROXY, "n_train": N_TRAIN, "n_eval": N_EVAL, "tokens": T,
        "batch": BATCH, "probe_sigma": PROBE_SIGMA, "n_probe": N_PROBE,
        "budgets": BUDGETS, "value_fns": value_fns, "eval_attacks": EVAL_ATTACKS,
        "headline_attack": EVAL_ATTACKS[0], "transfer_budget": TRANSFER_BUDGET,
        "oracle_attack": ORACLE_ATTACK,
        "eval_tokens": int(len(h_full.scores["token_difr"])),
        "batch_frac_of_null": float(BATCH / (0.5 * len(h_full.scores["token_difr"]))),
        "pareto": pareto, "transfer": transfer,
        "weights": head.weights(),
        "oracle_weights": oracle_head.weights(),
        "reliability": {
            "raw": {"conf": c_raw.tolist(), "acc": a_raw.tolist()},
            "sts": {"conf": c_sts.tolist(), "acc": a_sts.tolist()},
            "ece_raw": ece_raw, "ece_sts": ece_sts,
            "temperature": head.calibrator.temperature(),
            "bias": head.calibrator.bias.tolist(),
        },
    }
    out = RES_DIR / "confidence_head.json"
    out.write_text(json.dumps(payload, indent=2))

    # ---- 4. report + figures ---------------------------------------------
    print(f"\n{'budget':>8}" + "".join(f"{v:>13}" for v in value_fns))
    print("  " + "-" * (6 + 13 * len(value_fns)))
    for i, b in enumerate(BUDGETS):
        print(f"{b*100:>7.0f}%" + "".join(f"{pareto[EVAL_ATTACKS[0]][v][i]:>13.3f}"
                                          for v in value_fns))
    print(f"\nwrote {out}")
    plot_triage.main([])
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
