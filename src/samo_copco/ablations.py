"""Exposure and sensitivity ablation helpers for SAMO reader profiles."""

from __future__ import annotations

import pandas as pd

from .data_contracts import FORBIDDEN_PREDICTOR_COLUMNS
from .evaluation import run_lopo_evaluation


def feature_group_columns(frame: pd.DataFrame, group: str) -> list[str]:
    exposure = [
        column
        for column in frame.columns
        if column.startswith("lm_surprisal_exposure_") or column.startswith("lm_entropy_exposure_")
    ]
    sensitivity = [
        column
        for column in frame.columns
        if column.startswith("sensitivity__") and (column.endswith("__surprisal") or column.endswith("__entropy"))
    ]
    residual = [column for column in frame.columns if column.startswith("residual_mean__") or column.startswith("residual_std__")]
    groups = {
        "lm_exposure_only": exposure,
        "lm_sensitivity_only": sensitivity,
        "lm_residualized_profile": residual + sensitivity,
        "lm_exposure_plus_sensitivity": exposure + sensitivity,
    }
    if group not in groups:
        raise ValueError("unknown ablation group: " + group)
    selected = [column for column in groups[group] if column in frame.columns and column not in FORBIDDEN_PREDICTOR_COLUMNS]
    blocked = sorted(set(selected) & FORBIDDEN_PREDICTOR_COLUMNS)
    if blocked:
        raise ValueError("forbidden ablation columns selected: " + ", ".join(blocked))
    if not selected:
        raise ValueError("ablation group has no available columns: " + group)
    return selected


def ablation_definitions(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        group: feature_group_columns(frame, group)
        for group in ("lm_exposure_only", "lm_sensitivity_only", "lm_residualized_profile", "lm_exposure_plus_sensitivity")
    }


def run_ablation_suite(word_features: pd.DataFrame, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    base = run_lopo_evaluation(word_features, labels)
    profiles = base["reader_profiles"]
    rows: list[dict[str, object]] = []
    for name, columns in ablation_definitions(profiles).items():
        result = run_lopo_evaluation(word_features, labels, model_columns=columns)
        metric_row = dict(result["metrics"])
        metric_row["ablation"] = name
        metric_row["feature_count"] = len(columns)
        rows.append(metric_row)
    return pd.DataFrame(rows)
