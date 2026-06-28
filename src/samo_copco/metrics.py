"""Reader-level classification metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _as_arrays(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if y.shape[0] != score.shape[0]:
        raise ValueError("metric inputs must have equal length")
    if y.shape[0] == 0:
        raise ValueError("metric inputs are empty")
    return y, score


def roc_auc(y_true: Any, y_score: Any) -> float:
    y, score = _as_arrays(y_true, y_score)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    wins = 0.0
    for p_value in pos:
        wins += float((p_value > neg).sum())
        wins += 0.5 * float((p_value == neg).sum())
    return float(wins / (len(pos) * len(neg)))


def average_precision(y_true: Any, y_score: Any) -> float:
    y, score = _as_arrays(y_true, y_score)
    positives = int((y == 1).sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-score)
    ranked = y[order]
    hits = 0
    total = 0.0
    for idx, value in enumerate(ranked, start=1):
        if int(value) == 1:
            hits += 1
            total += hits / idx
    return float(total / positives)


def classification_metrics(y_true: Any, y_score: Any, *, threshold: float = 0.5) -> dict[str, float | int]:
    y, score = _as_arrays(y_true, y_score)
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pos = tp + fn
    neg = tn + fp
    tpr = tp / pos if pos else math.nan
    tnr = tn / neg if neg else math.nan
    precision_pos = tp / (tp + fp) if (tp + fp) else 0.0
    recall_pos = tpr if not math.isnan(tpr) else 0.0
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) else 0.0
    precision_neg = tn / (tn + fn) if (tn + fn) else 0.0
    recall_neg = tnr if not math.isnan(tnr) else 0.0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) else 0.0
    return {
        "n_readers": int(len(y)),
        "roc_auc": roc_auc(y, score),
        "average_precision": average_precision(y, score),
        "balanced_accuracy": float((tpr + tnr) / 2) if not (math.isnan(tpr) or math.isnan(tnr)) else math.nan,
        "accuracy": float((pred == y).mean()),
        "macro_f1": float((f1_pos + f1_neg) / 2),
        "brier_score": float(np.mean((score - y) ** 2)),
        "threshold": float(threshold),
    }
