"""Ablation helpers for SAMO reader profiles."""

from __future__ import annotations

import pandas as pd

from .evaluation import run_lopo_evaluation

DEFAULT_ABLATIONS = {
    "residual_profile": ["gaze_residual_mean", "gaze_residual_std", "predictability_residual_slope", "length_residual_slope"],
    "aggregate_gaze": ["gaze_ms_mean", "n_words"],
    "predictability_summary": ["predictability_mean", "predictability_std"],
}


def run_ablation_suite(word_features: pd.DataFrame, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, columns in DEFAULT_ABLATIONS.items():
        result = run_lopo_evaluation(word_features, labels, model_columns=columns)
        metric_row = dict(result["metrics"])
        metric_row["ablation"] = name
        metric_row["feature_count"] = len(columns)
        rows.append(metric_row)
    return pd.DataFrame(rows)
