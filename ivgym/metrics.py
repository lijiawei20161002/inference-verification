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


def tpr_at_fpr(neg: np.ndarray, pos: np.ndarray, fpr: float = 0.01) -> float:
    """True-positive rate at a target false-positive rate."""
    neg = np.asarray(neg, float)
    pos = np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return 0.0
    thresh = np.quantile(neg, 1.0 - fpr)
    return float(np.mean(pos > thresh))


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
