"""Fold-local residualization for SAMO gaze outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .data_contracts import GAZE_MS, LABEL, PREDICTABILITY, READER_ID


def default_residual_covariates(frame: pd.DataFrame, allow_proxy: bool = False) -> list[str]:
    """Choose residualization covariates, preferring separate LM surprisal and entropy."""

    covariates: list[str] = []
    if {"lm_word_surprisal", "lm_word_entropy"} <= set(frame.columns):
        covariates.extend(["lm_word_surprisal", "lm_word_entropy"])
    elif allow_proxy and PREDICTABILITY in frame.columns:
        covariates.append(PREDICTABILITY)
    else:
        raise ValueError("residualization requires lm_word_surprisal and lm_word_entropy")
    for column in ("word_length", "word_position"):
        if column in frame.columns:
            covariates.append(column)
    return covariates


@dataclass
class GazeResidualizer:
    outcome: str
    covariates: tuple[str, ...]
    coefficients_: np.ndarray | None = field(default=None, init=False)
    fill_values_: dict[str, float] = field(default_factory=dict, init=False)
    fit_reader_ids_: set[str] = field(default_factory=set, init=False)

    def fit(self, frame: pd.DataFrame, reader_column: str = READER_ID, heldout_readers: set[str] | None = None) -> "GazeResidualizer":
        if heldout_readers:
            overlap = set(frame[reader_column].astype(str)) & set(heldout_readers)
            if overlap:
                raise ValueError("held-out readers present during residualizer fit")
        missing = [column for column in (self.outcome, *self.covariates, reader_column) if column not in frame.columns]
        if missing:
            raise ValueError("missing residualization columns: " + ", ".join(missing))
        self.fit_reader_ids_ = set(frame[reader_column].astype(str).unique())
        x_parts = []
        self.fill_values_ = {}
        for column in self.covariates:
            values = pd.to_numeric(frame[column], errors="coerce")
            fill = float(values.mean()) if not values.dropna().empty else 0.0
            self.fill_values_[column] = fill
            x_parts.append(values.fillna(fill).to_numpy(dtype=float))
        design = np.column_stack([np.ones(len(frame)), *x_parts])
        y = pd.to_numeric(frame[self.outcome], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            raise ValueError(f"{self.outcome} contains non-numeric values")
        self.coefficients_ = np.linalg.pinv(design) @ y
        return self

    def transform(self, frame: pd.DataFrame) -> pd.Series:
        if self.coefficients_ is None:
            raise RuntimeError("residualizer must be fit before transform")
        x_parts = []
        for column in self.covariates:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(self.fill_values_.get(column, 0.0))
            x_parts.append(values.to_numpy(dtype=float))
        design = np.column_stack([np.ones(len(frame)), *x_parts])
        predicted = design @ self.coefficients_
        observed = pd.to_numeric(frame[self.outcome], errors="coerce").to_numpy(dtype=float)
        return pd.Series(observed - predicted, index=frame.index)


def fit_gaze_residualizers(train_frame: pd.DataFrame, gaze_columns: Iterable[str], covariates: Iterable[str]) -> dict[str, GazeResidualizer]:
    return {
        gaze: GazeResidualizer(gaze, tuple(covariates)).fit(train_frame)
        for gaze in gaze_columns
    }


def apply_gaze_residualizers(frame: pd.DataFrame, residualizers: dict[str, GazeResidualizer]) -> pd.DataFrame:
    out = frame.copy()
    for gaze, residualizer in residualizers.items():
        out[f"resid__{gaze}"] = residualizer.transform(out)
    return out


def crossfit_residualize_by_reader(
    frame: pd.DataFrame,
    reader_column: str = READER_ID,
    gaze_columns: Iterable[str] = (GAZE_MS,),
    covariates: Iterable[str] | None = None,
    allow_proxy: bool = False,
) -> pd.DataFrame:
    """Fit residualizers without each held-out reader and transform that reader."""

    if reader_column not in frame.columns:
        raise ValueError(f"missing reader column: {reader_column}")
    covariate_list = list(covariates) if covariates is not None else default_residual_covariates(frame, allow_proxy=allow_proxy)
    forbidden = {reader_column, LABEL, "target", "speech_id", "paragraph_id", "sentence_id", "text_id", "source_word_id", "lm_stable_word_id", "word_id"}
    blocked = sorted(set(covariate_list) & forbidden)
    if blocked:
        raise ValueError("forbidden residualization covariates: " + ", ".join(blocked))
    frames: list[pd.DataFrame] = []
    readers = sorted(frame[reader_column].astype(str).unique())
    for reader in readers:
        train = frame[frame[reader_column].astype(str) != reader].copy()
        test = frame[frame[reader_column].astype(str) == reader].copy()
        residualizers: dict[str, GazeResidualizer] = {}
        for gaze in gaze_columns:
            residualizer = GazeResidualizer(gaze, tuple(covariate_list))
            residualizer.fit(train, reader_column=reader_column, heldout_readers={reader})
            residualizers[gaze] = residualizer
        transformed = apply_gaze_residualizers(test, residualizers)
        transformed["residualizer_train_readers"] = ",".join(sorted(set(train[reader_column].astype(str))))
        frames.append(transformed)
    return pd.concat(frames, ignore_index=True)


@dataclass
class FoldLocalResidualizer:
    """Compatibility wrapper for one-outcome residualization."""

    outcome: str = GAZE_MS
    covariates: tuple[str, ...] = ("lm_word_surprisal", "lm_word_entropy", "word_length", "word_position")
    model_: GazeResidualizer | None = field(default=None, init=False)
    fit_reader_ids_: set[str] = field(default_factory=set, init=False)

    def fit(self, frame: pd.DataFrame, *, heldout_readers: set[str] | None = None) -> "FoldLocalResidualizer":
        self.model_ = GazeResidualizer(self.outcome, self.covariates).fit(frame, heldout_readers=heldout_readers)
        self.fit_reader_ids_ = set(self.model_.fit_reader_ids_)
        return self

    def transform(self, frame: pd.DataFrame, *, output_column: str = "gaze_residual") -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("residualizer must be fit before transform")
        out = frame.copy()
        out[output_column] = self.model_.transform(frame)
        return out
