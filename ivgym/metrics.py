"""ROC metrics in pure numpy (no sklearn dependency)."""
from __future__ import annotations

import numpy as np


def roc_auc(neg: np.ndarray, pos: np.ndarray) -> float:
    """AUC via the Mann-Whitney U statistic. pos = scores from misconfigured
    (should be higher), neg = scores from honest."""
    neg = np.asarray(neg, float)
    pos = np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return 0.5
    all_scores = np.concatenate([neg, pos])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty(len(all_scores))
    ranks[order] = np.arange(1, len(all_scores) + 1)
    # average ranks for ties
    _assign_tie_ranks(all_scores, ranks)
    r_pos = ranks[len(neg):].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def tpr_at_fpr(neg: np.ndarray, pos: np.ndarray, fpr: float = 0.005) -> float:
    """True-positive rate at a target false-positive rate.

    The threshold is the ``(1 - fpr)`` empirical quantile of ``neg``; the returned
    TPR is the fraction of ``pos`` strictly above it. For an *out-of-sample*
    operating point pass an independent calibration set as ``neg`` and the eval
    positives as ``pos`` (see ``harness.evaluate``)."""
    neg = np.asarray(neg, float)
    pos = np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return 0.0
    thresh = np.quantile(neg, 1.0 - fpr)
    return float(np.mean(pos > thresh))


def roc_curve(neg: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical ROC as monotone ``(fpr, tpr)`` step-function vertices, including
    the endpoints ``(0, 0)`` and ``(1, 1)``. Ties across the neg/pos boundary are
    resolved by collapsing equal scores into a single threshold (so a tie
    contributes a diagonal segment, matching the Mann-Whitney AUC)."""
    neg = np.asarray(neg, float)
    pos = np.asarray(pos, float)
    n_neg, n_pos = len(neg), len(pos)
    scores = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    order = np.argsort(-scores, kind="mergesort")   # high score first
    scores, labels = scores[order], labels[order]
    # one ROC vertex per group of equal scores (a lower threshold admits the group)
    distinct = np.where(np.diff(scores) != 0)[0]
    idx = np.concatenate([distinct, [len(scores) - 1]])
    tps = np.cumsum(labels)[idx]
    fps = np.cumsum(1.0 - labels)[idx]
    tpr = np.concatenate([[0.0], tps / n_pos]) if n_pos else np.zeros(len(idx) + 1)
    fpr = np.concatenate([[0.0], fps / n_neg]) if n_neg else np.zeros(len(idx) + 1)
    return fpr, tpr


def partial_auc(neg: np.ndarray, pos: np.ndarray, max_fpr: float = 0.005,
                standardized: bool = True) -> float:
    """Partial AUC over the low-false-positive region ``FPR in [0, max_fpr]`` --
    the operationally meaningful slice for verification, where an honest provider
    must almost never be flagged.

    The raw partial area is the integral of TPR over that FPR interval (with the
    ROC linearly interpolated at the ``max_fpr`` boundary). By default it is
    **McClish-standardized** so the scale matches full AUC: a random verifier
    scores 0.5 and a perfect one scores 1.0, regardless of ``max_fpr``::

        std = 0.5 * (1 + (pauc - min) / (max - min)),  min = max_fpr^2 / 2,  max = max_fpr

    Pass ``standardized=False`` for the raw normalized area (mean TPR over the
    region, in ``[0, 1]``). Returns 0.5 (standardized) / 0.0 (raw) on an empty
    input, matching the chance/degenerate convention of the other metrics."""
    neg = np.asarray(neg, float)
    pos = np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0 or max_fpr <= 0.0:
        return 0.5 if standardized else 0.0
    max_fpr = min(max_fpr, 1.0)
    fpr, tpr = roc_curve(neg, pos)
    # Clip the curve to [0, max_fpr], interpolating TPR at the boundary so the
    # partial area is exact rather than snapped to the nearest honest quantile.
    stop = int(np.searchsorted(fpr, max_fpr, side="right"))
    fpr_c = fpr[:stop]
    tpr_c = tpr[:stop]
    if fpr_c[-1] < max_fpr and stop < len(fpr):
        tpr_edge = np.interp(max_fpr, fpr, tpr)
        fpr_c = np.append(fpr_c, max_fpr)
        tpr_c = np.append(tpr_c, tpr_edge)
    pauc = float(np.trapz(tpr_c, fpr_c))            # raw area, in [0, max_fpr]
    if not standardized:
        return pauc / max_fpr
    min_area = 0.5 * max_fpr * max_fpr              # area under the chance diagonal
    max_area = max_fpr                              # perfect verifier
    return float(0.5 * (1.0 + (pauc - min_area) / (max_area - min_area)))


def _assign_tie_ranks(scores: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for t in range(i, j + 1):
                ranks[order[t]] = avg
        i = j + 1
