"""Fold-local residualization for reader-profile features."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_contracts import GAZE_LOG, READER_ID


@dataclass
class FoldLocalResidualizer:
    """Linear residualizer fit on training-reader rows only."""

    outcome: str = GAZE_LOG
    covariates: tuple[str, ...] = ("predictability_score", "word_length", "word_position")
    coefficients_: np.ndarray | None = field(default=None, init=False)
    fill_values_: dict[str, float] = field(default_factory=dict, init=False)
    fit_reader_ids_: set[str] = field(default_factory=set, init=False)

    def fit(self, frame: pd.DataFrame, *, heldout_readers: set[str] | None = None) -> "FoldLocalResidualizer":
        if heldout_readers:
            overlap = set(frame[READER_ID].astype(str)) & set(heldout_readers)
            if overlap:
                raise ValueError("held-out readers present during residualizer fit: " + ", ".join(sorted(overlap)))
        missing = [column for column in (self.outcome, *self.covariates, READER_ID) if column not in frame.columns]
        if missing:
            raise ValueError("missing residualization columns: " + ", ".join(missing))
        train = frame.copy()
        self.fit_reader_ids_ = set(train[READER_ID].astype(str).unique())
        x_parts = []
        self.fill_values_ = {}
        for column in self.covariates:
            series = pd.to_numeric(train[column], errors="coerce")
            fill = float(series.mean()) if not series.dropna().empty else 0.0
            self.fill_values_[column] = fill
            x_parts.append(series.fillna(fill).to_numpy(dtype=float))
        design = np.column_stack([np.ones(len(train)), *x_parts])
        y = pd.to_numeric(train[self.outcome], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            raise ValueError(f"{self.outcome} contains non-numeric values")
        self.coefficients_ = np.linalg.pinv(design) @ y
        return self

    def transform(self, frame: pd.DataFrame, *, output_column: str = "gaze_residual") -> pd.DataFrame:
        if self.coefficients_ is None:
            raise RuntimeError("residualizer must be fit before transform")
        missing = [column for column in (self.outcome, *self.covariates) if column not in frame.columns]
        if missing:
            raise ValueError("missing residualization columns: " + ", ".join(missing))
        out = frame.copy()
        x_parts = []
        for column in self.covariates:
            series = pd.to_numeric(out[column], errors="coerce").fillna(self.fill_values_.get(column, 0.0))
            x_parts.append(series.to_numpy(dtype=float))
        design = np.column_stack([np.ones(len(out)), *x_parts])
        predicted = design @ self.coefficients_
        observed = pd.to_numeric(out[self.outcome], errors="coerce").to_numpy(dtype=float)
        out[output_column] = observed - predicted
        return out


def fit_transform_fold_local(
    train_rows: pd.DataFrame,
    all_rows: pd.DataFrame,
    *,
    heldout_readers: set[str],
    outcome: str = GAZE_LOG,
    covariates: tuple[str, ...] = ("predictability_score", "word_length", "word_position"),
) -> tuple[pd.DataFrame, FoldLocalResidualizer]:
    residualizer = FoldLocalResidualizer(outcome=outcome, covariates=covariates)
    residualizer.fit(train_rows, heldout_readers=heldout_readers)
    return residualizer.transform(all_rows), residualizer
