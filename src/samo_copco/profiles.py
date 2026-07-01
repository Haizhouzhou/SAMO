"""SAMO reader-profile aggregation."""

from __future__ import annotations

import pandas as pd

from .data_contracts import GAZE_MS, LABEL, PREDICTABILITY, READER_ID


def _std(values: pd.Series) -> float:
    return float(values.std(ddof=0)) if len(values) > 1 else 0.0


def build_reader_profiles(frame: pd.DataFrame, *, reader_column: str = READER_ID) -> pd.DataFrame:
    """Build one residualized LM exposure/sensitivity profile per reader."""

    if reader_column not in frame.columns:
        raise ValueError(f"missing reader column: {reader_column}")
    residual_columns = [column for column in frame.columns if column.startswith("resid__")]
    if not residual_columns and "gaze_residual" in frame.columns:
        residual_columns = ["gaze_residual"]
    rows: list[dict[str, float | int | str]] = []
    for reader, group in frame.groupby(reader_column, sort=True):
        row: dict[str, float | int | str] = {READER_ID: str(reader), "n_words": int(len(group))}
        if LABEL in group.columns:
            labels = sorted(set(pd.to_numeric(group[LABEL], errors="coerce").dropna().astype(int).tolist()))
            if len(labels) != 1:
                raise ValueError(f"reader {reader} has inconsistent labels")
            row[LABEL] = int(labels[0])
        if GAZE_MS in group.columns:
            row["gaze_ms_mean"] = float(pd.to_numeric(group[GAZE_MS], errors="coerce").mean())
        if {"lm_word_surprisal", "lm_word_entropy"} <= set(group.columns):
            surprisal = pd.to_numeric(group["lm_word_surprisal"], errors="coerce")
            entropy = pd.to_numeric(group["lm_word_entropy"], errors="coerce")
            row["lm_surprisal_exposure_mean"] = float(surprisal.mean())
            row["lm_surprisal_exposure_std"] = _std(surprisal)
            row["lm_entropy_exposure_mean"] = float(entropy.mean())
            row["lm_entropy_exposure_std"] = _std(entropy)
            for residual_column in residual_columns:
                residual = pd.to_numeric(group[residual_column], errors="coerce")
                clean_name = residual_column.removeprefix("resid__")
                row[f"residual_mean__{clean_name}"] = float(residual.mean())
                row[f"residual_std__{clean_name}"] = _std(residual)
                row[f"sensitivity__{clean_name}__surprisal"] = float((residual * surprisal).mean())
                row[f"sensitivity__{clean_name}__entropy"] = float((residual * entropy).mean())
        if PREDICTABILITY in group.columns:
            pred = pd.to_numeric(group[PREDICTABILITY], errors="coerce")
            row["predictability_mean"] = float(pred.mean())
            row["predictability_std"] = _std(pred)
        rows.append(row)
    profiles = pd.DataFrame(rows).sort_values(READER_ID).reset_index(drop=True)
    if profiles[READER_ID].duplicated().any():
        raise ValueError("reader-profile aggregation produced duplicate readers")
    return profiles
