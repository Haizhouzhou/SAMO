"""Reader-level classifiers and predictor selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data_contracts import FORBIDDEN_PREDICTOR_COLUMNS, LABEL


def select_predictor_columns(frame: pd.DataFrame, candidates: Iterable[str] | None = None) -> list[str]:
    source = list(candidates) if candidates is not None else list(frame.columns)
    forbidden = set(FORBIDDEN_PREDICTOR_COLUMNS)
    selected: list[str] = []
    for column in source:
        if column in forbidden or column not in frame.columns:
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
class ReaderLogisticClassifier:
    """Logistic regression wrapper with deterministic preprocessing."""

    columns_: list[str] = field(default_factory=list, init=False)
    mean_: pd.Series | None = field(default=None, init=False)
    scale_: pd.Series | None = field(default=None, init=False)
    model_: LogisticRegression | None = field(default=None, init=False)

    def fit(self, frame: pd.DataFrame, label_column: str = LABEL, columns: Iterable[str] | None = None) -> "ReaderLogisticClassifier":
        self.columns_ = select_predictor_columns(frame, columns)
        y = pd.to_numeric(frame[label_column], errors="coerce").astype(int)
        if y.nunique() < 2:
            raise ValueError("training fold must contain both reader classes")
        x = frame[self.columns_].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self.mean_ = x.mean()
        self.scale_ = x.std(ddof=0).replace(0.0, 1.0)
        z = (x - self.mean_) / self.scale_
        self.model_ = LogisticRegression(max_iter=1000, solver="liblinear", random_state=17).fit(z, y)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("model must be fit before prediction")
        x = frame[self.columns_].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        z = (x - self.mean_) / self.scale_
        return self.model_.predict_proba(z)[:, 1]


RidgeLinearClassifier = ReaderLogisticClassifier
