"""Replay recorded clock evidence on a CPU; no model or API key required.

Run from a clone after `python -m pip install -e .`:
    python examples/quickstart.py

This scores existing measurements. It does not collect live evidence or fit a
calibration, and it must not be interpreted as a fresh provider verdict.
"""

import json
from pathlib import Path

from ivgym.clock import ClockSlope


def main():
    source = (
        Path(__file__).resolve().parents[1]
        / "docs/results/slope_verifier_window.json"
    )
    artifact = json.loads(source.read_text())
    cells = {cell["label"]: cell for cell in artifact["cells"]}
    intervals = 64  # Same slice used by the committed service integration check.
    short = cells["probe_lo"]["itl_ms"][:intervals]
    verifier = ClockSlope()

    print("IVGym | replay recorded H100 timing evidence")
    print(
        f"Model: {cells['honest_hi']['model']} | "
        f"claimed context: {artifact['claimed_ctx']} tokens"
    )
    print(f"{intervals} paired intervals per condition; device timings, not network arrivals.")
    print(f"\n{'condition':<22} {'long - short (ms)':>17} {'clock score':>15}")
    for key, label in (
        ("honest_hi", "honest full context"),
        ("window_512", "512-token window"),
    ):
        score = float(verifier.score_pairs(short, cells[key]["itl_ms"][:intervals]).mean())
        print(f"{label:<22} {-score:>17.3f} {score:>15.3f}")
    print("\nA smaller timing gap raises the anomaly score.")
    print("Scores only: no calibrated verdict or live provider check in this demo.")


if __name__ == "__main__":
    main()
