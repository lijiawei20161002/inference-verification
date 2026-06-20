"""A single-run verification view, in the style of a `verify.py` audit.

Where `experiments.run` prints an attack x defense AUC *scoreboard*, this prints
the verifier's-eye view of *one* claimed run: it scores every claimed token with
one defense, calibrates two thresholds on honest tokens, and classifies each
token as SAFE / SUSPICIOUS / DANGEROUS -- rendering the token stream colorized
and a results summary, then the batch-level verdict (S vs tau).

    python -m experiments.verify_demo --claimed honest
    python -m experiments.verify_demo --claimed quant_4bit
    python -m experiments.verify_demo --claimed adv_quant_temp --tokens 96
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ivgym import attacks, defenses, harness
from ivgym.backends import make_backend
from ivgym.core import SamplingSpec

# ANSI
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, RED, CYAN, GREY = (
    "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[90m")

# A pool of plausible "generated" words, so the colored stream reads like prose
# instead of integer token ids. Coloring is driven by the real per-token score;
# only the glyph is cosmetic.
_WORDS = (
    "the model returns a stream of tokens that look entirely ordinary at first "
    "glance yet every one of them carries a divergence score the verifier "
    "recomputes under the agreed sampling spec and compares against what an "
    "honest provider would have produced on the same prompt with the same seed "
    "so a cheaper quantized run a wrong temperature or a spiked logit leaves a "
    "trail of small statistical tells that quietly accumulate across the batch"
).split()


def _classify(scores, tau_susp, tau_dang):
    safe = scores < tau_susp
    dang = scores >= tau_dang
    susp = (~safe) & (~dang)
    return safe, susp, dang


def _render_stream(scores, tau_susp, tau_dang, width=84, max_tokens=108):
    safe, susp, dang = _classify(scores, tau_susp, tau_dang)
    out, line = [], ""
    for i in range(min(len(scores), max_tokens)):
        w = _WORDS[i % len(_WORDS)]
        color = RED if dang[i] else (YELLOW if susp[i] else GREEN)
        piece = f"{color}{w}{RESET}"
        if len(line) - line.count("\033") * 5 + len(w) + 1 > width:
            out.append("  " + line)
            line = ""
        line += (" " if line else "") + piece
    if line:
        out.append("  " + line)
    if len(scores) > max_tokens:
        out.append(f"  {DIM}... {len(scores) - max_tokens} more tokens{RESET}")
    return "\n".join(out)


def _bar(frac, width=24):
    fill = int(round(frac * width))
    return "█" * fill + "·" * (width - fill)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="experiments.verify_demo")
    ap.add_argument("--claimed", default="quant_4bit",
                    help="Attack name the provider claims-or-cheats with (default: quant_4bit).")
    ap.add_argument("--defense", default="token_difr")
    ap.add_argument("--prompts", type=int, default=48)
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--vocab", type=int, default=512)
    args = ap.parse_args(argv)

    backend = make_backend("synthetic", vocab=args.vocab)
    spec = SamplingSpec()
    defn = defenses.get(args.defense)

    # 1. Calibrate the verifier on honest tokens.
    #    Per-token gates -> the SUSPICIOUS / DANGEROUS labels on the stream.
    #    Batch gate     -> the verdict, set on the honest *batch-mean* spread
    #                      at a 1% false-positive budget (a batch mean is far
    #                      tighter than a single token, so it needs its own tau).
    honest_seqs = harness.generate_dataset(
        backend, attacks.get("honest"), spec, args.prompts, args.tokens)
    honest = harness.verify(backend, honest_seqs, spec, [defn]).scores[defn.name]
    tau_susp = float(np.percentile(honest, 99.0))     # SUSPICIOUS gate
    tau_dang = float(np.percentile(honest, 99.9))     # DANGEROUS gate

    rng = np.random.default_rng(0)
    b = min(256, len(honest) // 2)
    honest_batches = harness.batch_means(honest, b, 400, rng)
    tau_batch = float(np.percentile(honest_batches, 99.0))   # 1% FPR

    # 2. Verify the claimed run.
    claim_seqs = harness.generate_dataset(
        backend, attacks.get(args.claimed), spec, args.prompts, args.tokens)
    scores = harness.verify(backend, claim_seqs, spec, [defn]).scores[defn.name]

    safe, susp, dang = _classify(scores, tau_susp, tau_dang)
    n = len(scores)
    n_safe, n_susp, n_dang = int(safe.sum()), int(susp.sum()), int(dang.sum())
    S = float(harness.batch_means(scores, b, 400, rng).mean())  # batch statistic
    flagged = S > tau_batch

    # 3. Render.
    print()
    print(f"  {BOLD}{CYAN}ivgym{RESET} · inference verification")
    print(f"  {DIM}spec{RESET}      temperature={spec.temperature}  "
          f"top_k={spec.top_k}  top_p={spec.top_p}  seed={spec.seed}")
    print(f"  {DIM}defense{RESET}   {defn.name}   "
          f"{DIM}calibrated on {len(honest)} honest tokens{RESET}")
    print(f"  {DIM}gates{RESET}     {YELLOW}suspicious{RESET} > {tau_susp:5.2f}    "
          f"{RED}dangerous{RESET} > {tau_dang:5.2f}   "
          f"{DIM}(99th / 99.9th honest pct){RESET}")
    print()
    print(f"  verifying claimed={BOLD}{args.claimed!r}{RESET}  ({n} tokens)\n")
    print(_render_stream(scores, tau_susp, tau_dang))
    print()
    print(f"  {BOLD}Results:{RESET}")
    print(f"    Total tokens:      {n:>6}")
    print(f"    {GREEN}Safe{RESET} tokens:       {n_safe:>6} ({100*n_safe/n:5.2f}%)  "
          f"{GREEN}{_bar(n_safe/n)}{RESET}")
    print(f"    {YELLOW}Suspicious{RESET} tokens: {n_susp:>6} ({100*n_susp/n:5.2f}%)  "
          f"{YELLOW}{_bar(n_susp/n)}{RESET}")
    print(f"    {RED}Dangerous{RESET} tokens:  {n_dang:>6} ({100*n_dang/n:5.2f}%)  "
          f"{RED}{_bar(n_dang/n)}{RESET}")
    print()
    if flagged:
        print(f"  {BOLD}{RED}● DEVIATION DETECTED{RESET}  "
              f"batch S={S:.2f} > τ={tau_batch:.2f}   {DIM}provider ≠ spec{RESET}")
    else:
        print(f"  {BOLD}{GREEN}● CONSISTENT WITH SPEC{RESET}  "
              f"batch S={S:.2f} ≤ τ={tau_batch:.2f}   {DIM}honest provider{RESET}")
    print()


if __name__ == "__main__":
    main()
