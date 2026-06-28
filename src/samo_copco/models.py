"""Small deterministic reader-level classifier used by SAMO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .data_contracts import FORBIDDEN_PREDICTOR_COLUMNS, LABEL


def select_predictor_columns(frame: pd.DataFrame, candidates: Iterable[str] | None = None) -> list[str]:
    source = list(candidates) if candidates is not None else list(frame.columns)
    selected: list[str] = []
    forbidden = set(FORBIDDEN_PREDICTOR_COLUMNS)
    for column in source:
        if column in forbidden:
            continue
        if column not in frame.columns:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            selected.append(column)
    blocked = sorted(set(selected) & forbidden)
    if blocked:
        raise ValueError("forbidden predictors selected: " + ", ".join(blocked))
    if not selected:
        raise ValueError("no numeric non-identifier predictors available")
    return selected


@dataclass
class RidgeLinearClassifier:
    """Ridge-stabilized linear probability scorer with sigmoid output."""

    alpha: float = 1.0
    columns_: list[str] = field(default_factory=list, init=False)
    mean_: np.ndarray | None = field(default=None, init=False)
    scale_: np.ndarray | None = field(default=None, init=False)
    coef_: np.ndarray | None = field(default=None, init=False)

    def fit(self, frame: pd.DataFrame, label_column: str = LABEL, columns: Iterable[str] | None = None) -> "RidgeLinearClassifier":
        self.columns_ = select_predictor_columns(frame, columns)
        y = pd.to_numeric(frame[label_column], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            raise ValueError("labels contain non-numeric values")
        if len(set(y.astype(int).tolist())) < 2:
            raise ValueError("training fold must contain both reader classes")
        x = frame[self.columns_].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        z = (x - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(z)), z])
        target = np.where(y > 0, 1.0, -1.0)
        penalty = np.eye(design.shape[1]) * float(self.alpha)
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(design.T @ design + penalty) @ design.T @ target
        return self

    def decision_function(self, frame: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("model must be fit before prediction")
        x = frame[self.columns_].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        z = (x - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(z)), z])
        return design @ self.coef_

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        score = self.decision_function(frame)
        return 1.0 / (1.0 + np.exp(-score))
