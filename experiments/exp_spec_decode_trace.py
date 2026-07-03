"""Speculative-decoding **trace verification** sweep -- cheat x check detection AUC.

Unlike the DiFR experiments this runs on **CPU with pure numpy** (no GPU, no model
download): the provider's speculative-decoding step is simulated over synthetic
correlated (target, draft) distributions (`ivgym.spec_decode.synthetic_positions`),
so the whole client-side verifier is exercised end-to-end here. A real backend
would feed actual target/draft logprobs from vLLM's trace instead; the checks and
the verifier are unchanged.

For each cheating provider we report the detection AUC of every client-side check
(honest traces vs cheat traces). The headline structure:

  * sampler-level cheats (over-accept, coin-fudge, skip-residual) are caught by the
    **cheap, no-recompute** self-consistency checks;
  * `draft_as_target` -- run only the draft model and relabel it as the target --
    produces a perfectly self-consistent trace and EVADES every no-recompute check;
    only `target_spotcheck` (which recomputes a subset of true target logprobs)
    catches it. That is the DiFR "recomputation is necessary" result, now for the
    speculative-decoding trace.

Run:  python -m experiments.exp_spec_decode_trace
Env overrides: IVGYM_TRACES, IVGYM_POS, IVGYM_VOCAB, IVGYM_AGREE, IVGYM_FPR.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import spec_decode as sd
from ivgym.core import SamplingSpec
from ivgym.metrics import roc_auc

N_TRACES = int(os.environ.get("IVGYM_TRACES", 80))
N_POS = int(os.environ.get("IVGYM_POS", 200))
VOCAB = int(os.environ.get("IVGYM_VOCAB", 64))
AGREE = float(os.environ.get("IVGYM_AGREE", 0.8))
FPR = float(os.environ.get("IVGYM_FPR", 0.01))

CHEATS = ["over_accept_naive", "over_accept_coinfudge", "skip_residual",
          "sampling_bug", "quant_target", "draft_as_target"]


def build(cheat_name: str, n_traces: int, seed0: int) -> list:
    """Generate `n_traces` traces from one provider over fresh synthetic positions."""
    spec = SamplingSpec()
    out = []
    for t in range(n_traces):
        rng = np.random.default_rng(1_000_003 * seed0 + t)
        pos = sd.synthetic_positions(rng, N_POS, vocab=VOCAB, agreement=AGREE)
        out.append(sd.generate_trace(rng, sd.get_cheat(cheat_name), pos, spec))
    return out


def check_scores(check: sd.Check, traces: list) -> np.ndarray:
    return np.array([
        check.score(t, sd.make_oracle(t) if check.needs_oracle else None)
        for t in traces], float)


def main():
    checks = list(sd.all_checks().values())

    # Two independent honest draws: a large one to calibrate the thresholds
    # (stable tail quantiles), a separate one as the detection null.
    honest_cal = build("honest", max(N_TRACES, 400), seed0=1)
    honest_null = build("honest", N_TRACES, seed0=2)

    print(f"traces={N_TRACES}  positions/trace={N_POS}  vocab={VOCAB}  "
          f"agreement={AGREE}  target-FPR={FPR}")
    acc = np.mean([np.mean([s.accepted for s in t.steps]) for t in honest_null])
    print(f"honest acceptance rate = {acc:.3f}  (= 1 - TV(target, draft))\n")

    # --- Detection-AUC grid: rows = cheat, cols = check ---
    names = [c.name for c in checks]
    w = max(len(n) for n in names) + 1
    header = f"{'cheat':>22} | " + " ".join(f"{n:>{w}}" for n in names)
    print("Detection AUC (honest vs cheat), higher = better caught")
    print(header)
    print("-" * len(header))
    honest_by_check = {c.name: check_scores(c, honest_null) for c in checks}
    caught_by = {}
    for cheat in CHEATS:
        traces = build(cheat, N_TRACES, seed0=3)
        cells = []
        winners = []
        for c in checks:
            auc = roc_auc(honest_by_check[c.name], check_scores(c, traces))
            cells.append(f"{auc:>{w}.3f}")
            if auc >= 0.9:
                winners.append(c.name)
        caught_by[cheat] = winners
        print(f"{cheat:>22} | " + " ".join(cells))

    # --- Single-trace client verdicts under the calibrated verifier ---
    print("\nClient-side verdicts (thresholds calibrated on honest, "
          f"FPR={FPR}):")
    full = sd.TraceVerifier(use_oracle=True).calibrate(honest_cal, fpr=FPR)
    cheap = sd.TraceVerifier(use_oracle=False).calibrate(honest_cal, fpr=FPR)

    # honest false-positive rate on the held-out null (sanity: should be ~FPR)
    fp = np.mean([full.verify(t).flagged for t in honest_null])
    print(f"  {'honest (null)':>22}  ->  flagged {fp*100:.1f}% of traces "
          f"(target {FPR*100:.1f}%)")
    for cheat in CHEATS:
        t = build(cheat, 1, seed0=7)[0]
        vf = full.verify(t)
        vc = cheap.verify(t)
        by = [c.check for c in vf.checks if c.flagged]
        print(f"  {cheat:>22}  ->  full={'FLAG' if vf.flagged else 'pass'}  "
              f"no-recompute={'FLAG' if vc.flagged else 'pass'}  "
              f"by={by}")

    print("\nTakeaway:")
    for cheat in CHEATS:
        print(f"  {cheat:<22} caught by: {caught_by[cheat] or ['<none>']}")
    print("  -> draft_as_target is invisible to every no-recompute check; only the\n"
          "     recompute spot-check sees it. Self-consistency is a cheap first line;\n"
          "     recomputation remains necessary for relabelling cheats.")


if __name__ == "__main__":
    main()
