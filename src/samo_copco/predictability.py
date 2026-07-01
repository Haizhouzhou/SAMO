"""Predictability and LM-feature merge helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_contracts import LM_STABLE_WORD_ID, PREDICTABILITY, WORD, WORD_LENGTH, WORD_POSITION
from .stimulus_reconstruction import make_lm_stable_word_id


def validate_predictability_columns(frame: pd.DataFrame, *, allow_synthetic_score: bool = False) -> None:
    """Validate that real LM columns or an explicit synthetic score are present."""

    if {"lm_word_surprisal", "lm_word_entropy"} <= set(frame.columns):
        for column in ("lm_word_surprisal", "lm_word_entropy"):
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any():
                raise ValueError(f"{column} contains non-numeric values")
        return
    if allow_synthetic_score and PREDICTABILITY in frame.columns:
        values = pd.to_numeric(frame[PREDICTABILITY], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{PREDICTABILITY} contains non-numeric values")
        return
    raise ValueError("LM surprisal and entropy columns are required for SAMO predictability features")


def require_lm_predictability_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in ("lm_word_surprisal", "lm_word_entropy") if column not in frame.columns]
    if missing:
        raise ValueError("missing LM predictability columns: " + ", ".join(missing))
    validate_predictability_columns(frame)


def merge_lm_features(word_rows: pd.DataFrame, lm_features: pd.DataFrame) -> pd.DataFrame:
    """Many-to-one merge from stimulus-level LM rows to repeated participant-word rows."""

    words = make_lm_stable_word_id(word_rows)
    lm = make_lm_stable_word_id(lm_features)
    if lm[LM_STABLE_WORD_ID].duplicated().any():
        raise ValueError("stimulus-level LM rows must be unique by lm_stable_word_id")
    merged = words.merge(lm, on=LM_STABLE_WORD_ID, how="left", validate="many_to_one", suffixes=("", "_lm"))
    for column in ("speech_id", "paragraph_id", "sentence_id", "source_word_id", WORD):
        lm_column = f"{column}_lm"
        if column not in merged.columns and lm_column in merged.columns:
            merged[column] = merged[lm_column]
    missing_lm = merged["lm_word_surprisal"].isna() if "lm_word_surprisal" in merged.columns else pd.Series(True, index=merged.index)
    if bool(missing_lm.any()):
        raise ValueError("LM merge left unmatched word rows")
    return merged[[column for column in merged.columns if not column.endswith("_lm")]].copy()


def add_synthetic_proxy_predictability(frame: pd.DataFrame) -> pd.DataFrame:
    """Create an explicit synthetic score for tests that do not use LM features."""

    out = frame.copy()
    if WORD_LENGTH not in out.columns:
        out[WORD_LENGTH] = out[WORD].astype(str).str.len().astype(float)
    if WORD_POSITION not in out.columns:
        out[WORD_POSITION] = range(1, len(out) + 1)
    length = pd.to_numeric(out[WORD_LENGTH], errors="coerce").fillna(out[WORD_LENGTH].median())
    position = pd.to_numeric(out[WORD_POSITION], errors="coerce").fillna(out[WORD_POSITION].median())
    length_scaled = (length - length.min()) / max(float(length.max() - length.min()), 1.0)
    position_scaled = (position - position.min()) / max(float(position.max() - position.min()), 1.0)
    out[PREDICTABILITY] = np.clip(1.0 - 0.65 * length_scaled - 0.20 * position_scaled, 0.02, 0.98)
    out["predictability_source"] = "explicit_synthetic"
    return out


def add_predictability_columns(frame: pd.DataFrame, *, allow_proxy: bool = False, proxy_source: str = "explicit_synthetic") -> pd.DataFrame:
    """Backward-compatible helper that does not create synthetic scores unless requested."""

    if {"lm_word_surprisal", "lm_word_entropy"} <= set(frame.columns):
        out = frame.copy()
        out[PREDICTABILITY] = np.exp(-pd.to_numeric(out["lm_word_surprisal"], errors="coerce")).clip(0.0, 1.0)
        out["predictability_source"] = "lm_word_surprisal"
        return out
    if PREDICTABILITY in frame.columns:
        validate_predictability_columns(frame, allow_synthetic_score=True)
        return frame.copy()
    if allow_proxy:
        out = add_synthetic_proxy_predictability(frame)
        out["predictability_source"] = proxy_source
        return out
    raise ValueError("predictability requires LM columns or explicit synthetic mode")
