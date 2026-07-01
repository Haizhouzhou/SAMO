"""Reader-level classification metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score


def _as_arrays(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if y.shape[0] != score.shape[0]:
        raise ValueError("metric inputs must have equal length")
    if y.shape[0] == 0:
        raise ValueError("metric inputs are empty")
    return y, score


def classification_metrics(y_true: Any, y_score: Any, *, threshold: float = 0.5) -> dict[str, float | int]:
    y, score = _as_arrays(y_true, y_score)
    pred = (score >= threshold).astype(int)
    roc = float(roc_auc_score(y, score)) if len(set(y.tolist())) == 2 else math.nan
    pr = float(average_precision_score(y, score)) if int((y == 1).sum()) else math.nan
    brier = float(brier_score_loss(y, score))
    return {
        "n_readers": int(len(y)),
        "roc_auc": roc,
        "pr_auc": pr,
        "average_precision": pr,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "brier_score": brier,
        "brier": brier,
        "threshold": float(threshold),
    }
