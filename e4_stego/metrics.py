"""Detection / capacity metrics. The verifier flags a token when its GLS is low, so we
treat (-GLS) as the detector score and sweep the threshold for ROC."""
from __future__ import annotations

import numpy as np


def bit_error_rate(sent: list[int], recovered: list[int]) -> tuple[float, int]:
    n = min(len(sent), len(recovered))
    if n == 0:
        return 0.0, 0
    s = np.asarray(sent[:n]); r = np.asarray(recovered[:n])
    errs = int((s != r).sum())
    # any length mismatch counts as errors too (desync / truncation)
    errs += abs(len(sent) - len(recovered))
    denom = max(len(sent), len(recovered))
    return errs / denom, errs


def roc_auc(benign_scores, attack_scores):
    """AUC for separating attack (positive) from benign (negative) using detector score
    = -GLS (higher => more attack-like). Returns (auc, fpr_grid, tpr_grid)."""
    benign = np.asarray(benign_scores, dtype=np.float64)
    attack = np.asarray(attack_scores, dtype=np.float64)
    y = np.concatenate([np.ones_like(attack), np.zeros_like(benign)])
    s = np.concatenate([attack, benign])
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    P = max(1, int((y == 1).sum())); N = max(1, int((y == 0).sum()))
    tps = np.cumsum(y == 1); fps = np.cumsum(y == 0)
    tpr = tps / P; fpr = fps / N
    tpr = np.concatenate([[0.0], tpr]); fpr = np.concatenate([[0.0], fpr])
    auc = float(np.trapz(tpr, fpr))
    return auc, fpr, tpr


def tpr_at_fpr(benign_scores, attack_scores, target_fpr: float = 0.01) -> float:
    auc, fpr, tpr = roc_auc(benign_scores, attack_scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, idx)
    return float(tpr[idx])


def throughput_projection(bits_per_token: float, tokens_per_sec: float):
    bytes_per_sec = bits_per_token * tokens_per_sec / 8.0
    gb_per_day = bytes_per_sec * 86400 / 1e9
    days_to_1tb = (1e12 / bytes_per_sec / 86400) if bytes_per_sec > 0 else float("inf")
    return {"bytes_per_sec": bytes_per_sec, "gb_per_day": gb_per_day, "days_to_1tb": days_to_1tb}
