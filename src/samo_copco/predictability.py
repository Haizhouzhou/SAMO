"""Predictability-like feature construction used by public SAMO commands."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_contracts import PREDICTABILITY, WORD, WORD_LENGTH, WORD_POSITION


def add_predictability_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with deterministic predictability-style columns.

    If a user-provided predictability column exists it is validated and preserved. If it is
    absent, a transparent lexical proxy is created from word length and within-text position.
    This fallback is for pipeline execution and synthetic diagnostics, not a language-model
    score claim.
    """

    out = frame.copy()
    if WORD_LENGTH not in out.columns:
        out[WORD_LENGTH] = out[WORD].astype(str).str.len().astype(float)
    if WORD_POSITION not in out.columns:
        out[WORD_POSITION] = out.groupby("text_id").cumcount().astype(float) + 1.0
    if PREDICTABILITY in out.columns:
        values = pd.to_numeric(out[PREDICTABILITY], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{PREDICTABILITY} contains non-numeric values")
        out[PREDICTABILITY] = values.astype(float)
        return out
    length = pd.to_numeric(out[WORD_LENGTH], errors="coerce").fillna(out[WORD_LENGTH].median())
    position = pd.to_numeric(out[WORD_POSITION], errors="coerce").fillna(out[WORD_POSITION].median())
    length_scaled = (length - length.min()) / max(float(length.max() - length.min()), 1.0)
    position_scaled = (position - position.min()) / max(float(position.max() - position.min()), 1.0)
    raw = 1.0 - 0.70 * length_scaled - 0.15 * position_scaled
    out[PREDICTABILITY] = np.clip(raw, 0.02, 0.98).astype(float)
    return out
