"""The context-slope clock verifier used in the timing experiments.

D = ITL(long context) - ITL(short context). A truncating provider reduces D,
so the anomaly score is -D. Timing units and capture provenance must match the
honest calibration; SSE event gaps are not automatically GPU token times.
"""

from __future__ import annotations
import numpy as np
from .verifiers import Verifier


class ClockSlope(Verifier):
    name = "clock_slope"
    value_fn = "uniform"

    @staticmethod
    def differences(short_ms, long_ms):
        short, long = np.asarray(short_ms, float), np.asarray(long_ms, float)
        if short.ndim != 1 or short.shape != long.shape or not short.size:
            raise ValueError(
                "clock probes require equally sized nonempty interval arrays"
            )
        if (
            not np.isfinite(short).all()
            or not np.isfinite(long).all()
            or np.any(short < 0)
            or np.any(long < 0)
        ):
            raise ValueError("clock intervals must be finite and nonnegative")
        return long - short

    def score_pairs(self, short_ms, long_ms):
        return -self.differences(short_ms, long_ms)
