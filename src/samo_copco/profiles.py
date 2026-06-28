"""Reader-profile aggregation for SAMO."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_contracts import GAZE_MS, LABEL, PREDICTABILITY, READER_ID, WORD_LENGTH


def _slope(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    x = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    centered = x - x.mean()
    denom = float(np.dot(centered, centered))
    if denom == 0.0:
        return 0.0
    return float(np.dot(centered, y - y.mean()) / denom)


def build_reader_profiles(frame: pd.DataFrame, *, residual_column: str = "gaze_residual") -> pd.DataFrame:
    required = {READER_ID, GAZE_MS, PREDICTABILITY, WORD_LENGTH, residual_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("missing profile columns: " + ", ".join(missing))
    rows: list[dict[str, float | int | str]] = []
    for reader_id, group in frame.groupby(READER_ID, sort=True):
        residual = pd.to_numeric(group[residual_column], errors="coerce")
        gaze = pd.to_numeric(group[GAZE_MS], errors="coerce")
        predictability = pd.to_numeric(group[PREDICTABILITY], errors="coerce")
        row: dict[str, float | int | str] = {
            READER_ID: str(reader_id),
            "n_words": int(len(group)),
            "gaze_residual_mean": float(residual.mean()),
            "gaze_residual_std": float(residual.std(ddof=0)) if len(group) > 1 else 0.0,
            "predictability_residual_slope": _slope(group, PREDICTABILITY, residual_column),
            "length_residual_slope": _slope(group, WORD_LENGTH, residual_column),
            "gaze_ms_mean": float(gaze.mean()),
            "predictability_mean": float(predictability.mean()),
            "predictability_std": float(predictability.std(ddof=0)) if len(group) > 1 else 0.0,
        }
        if LABEL in group.columns:
            unique_labels = sorted(set(pd.to_numeric(group[LABEL], errors="coerce").dropna().astype(int).tolist()))
            if len(unique_labels) != 1:
                raise ValueError(f"reader {reader_id} has inconsistent labels")
            row[LABEL] = int(unique_labels[0])
        rows.append(row)
    profiles = pd.DataFrame(rows).sort_values(READER_ID).reset_index(drop=True)
    if profiles[READER_ID].duplicated().any():
        raise ValueError("reader-profile aggregation produced duplicate readers")
    return profiles
