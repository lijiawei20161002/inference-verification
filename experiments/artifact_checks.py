"""Numerical checks for archived bootstrap statistics across NumPy versions."""
import math


def bootstrap_summary_matches(actual, expected):
    """Allow float64 roundoff while requiring the same fields and change count.

    Reduction and quantile implementations can differ in their last few bits.
    Hashes and raw same-shape scores must still be checked exactly by callers.
    """
    scalars = ('d_prime', 'base_mean', 'alt_mean', 'base_sd')
    keys = {*scalars, 'ci', 'frac_tokens_changed'}
    for row in (actual, expected):
        if not isinstance(row, dict) or set(row) != keys:
            return False
        if not isinstance(row['ci'], list) or len(row['ci']) != 2:
            return False
        values = [row[k] for k in scalars] + row['ci'] + [row['frac_tokens_changed']]
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in values):
            return False
        if not 0 <= row['frac_tokens_changed'] <= 1:
            return False
    if actual['frac_tokens_changed'] != expected['frac_tokens_changed']:
        return False
    actual_values = [actual[k] for k in scalars] + actual['ci']
    expected_values = [expected[k] for k in scalars] + expected['ci']
    return all(math.isclose(a, e, rel_tol=1e-12, abs_tol=1e-14)
               for a, e in zip(actual_values, expected_values))
