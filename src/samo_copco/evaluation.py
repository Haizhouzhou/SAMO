"""Reader-disjoint LOPO evaluation for SAMO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data import prepare_word_table, write_table
from .data_contracts import GAZE_MS, LABEL, READER_ID
from .metrics import classification_metrics
from .models import ReaderLogisticClassifier, select_predictor_columns
from .profiles import build_reader_profiles
from .residualization import crossfit_residualize_by_reader, default_residual_covariates


def make_lopo_splits(readers: Iterable[str]) -> list[dict[str, object]]:
    reader_list = sorted({str(reader) for reader in readers})
    if len(reader_list) < 3:
        raise ValueError("LOPO evaluation requires at least three readers")
    return [
        {
            "fold_id": fold_id,
            "train_readers": [reader for reader in reader_list if reader != heldout],
            "test_readers": [heldout],
            "heldout_reader": heldout,
        }
        for fold_id, heldout in enumerate(reader_list, start=1)
    ]


def assert_reader_disjoint(train_readers: Iterable[str], test_readers: Iterable[str]) -> None:
    overlap = set(map(str, train_readers)) & set(map(str, test_readers))
    if overlap:
        raise ValueError("reader overlap between train and test: " + ", ".join(sorted(overlap)))


def _profiles_for_fold(prepared: pd.DataFrame, train_readers: set[str], test_readers: set[str], allow_proxy: bool) -> pd.DataFrame:
    train_rows = prepared[prepared[READER_ID].astype(str).isin(train_readers)].copy()
    fold_rows = prepared[prepared[READER_ID].astype(str).isin(train_readers | test_readers)].copy()
    covariates = default_residual_covariates(prepared, allow_proxy=allow_proxy)
    frames: list[pd.DataFrame] = []
    for reader in sorted(train_readers | test_readers):
        test = fold_rows[fold_rows[READER_ID].astype(str) == reader].copy()
        if reader in test_readers:
            train = train_rows
        else:
            train = train_rows[train_rows[READER_ID].astype(str) != reader].copy()
        from .residualization import apply_gaze_residualizers, fit_gaze_residualizers

        residualizers = fit_gaze_residualizers(train, (GAZE_MS,), covariates)
        frames.append(apply_gaze_residualizers(test, residualizers))
    return build_reader_profiles(pd.concat(frames, ignore_index=True))


def run_lopo_evaluation(
    word_features: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    *,
    model_columns: Iterable[str] | None = None,
    allow_proxy: bool = False,
) -> dict[str, pd.DataFrame | dict[str, float | int]]:
    prepared = prepare_word_table(word_features, labels, allow_synthetic_proxy=allow_proxy)
    if LABEL not in prepared.columns:
        raise ValueError("reader labels are required for LOPO evaluation")
    readers = sorted(prepared[READER_ID].astype(str).unique())
    splits = make_lopo_splits(readers)
    prediction_rows: list[dict[str, object]] = []
    heldout_profiles: list[pd.DataFrame] = []
    skipped_folds = 0
    for split in splits:
        train_readers = set(map(str, split["train_readers"]))
        test_readers = set(map(str, split["test_readers"]))
        assert_reader_disjoint(train_readers, test_readers)
        profiles = _profiles_for_fold(prepared, train_readers, test_readers, allow_proxy)
        train_profiles = profiles[profiles[READER_ID].astype(str).isin(train_readers)].copy()
        test_profiles = profiles[profiles[READER_ID].astype(str).isin(test_readers)].copy()
        if train_profiles[LABEL].nunique() < 2:
            skipped_folds += 1
            continue
        columns = select_predictor_columns(train_profiles, model_columns)
        model = ReaderLogisticClassifier().fit(train_profiles, columns=columns)
        scores = model.predict_proba(test_profiles)
        for idx, (_, row) in enumerate(test_profiles.iterrows()):
            prediction_rows.append(
                {
                    "fold_id": int(split["fold_id"]),
                    READER_ID: str(row[READER_ID]),
                    "true_label": int(row[LABEL]),
                    "predicted_probability": float(scores[idx]),
                    "predicted_label": int(scores[idx] >= 0.5),
                    "model_feature_count": int(len(columns)),
                }
            )
        heldout = test_profiles.copy()
        heldout["fold_id"] = int(split["fold_id"])
        heldout_profiles.append(heldout)
    if not prediction_rows:
        raise ValueError("all LOPO folds were skipped")
    predictions = pd.DataFrame(prediction_rows).sort_values(["fold_id", READER_ID]).reset_index(drop=True)
    profile_frame = pd.concat(heldout_profiles, ignore_index=True).sort_values(READER_ID).reset_index(drop=True)
    metrics = classification_metrics(predictions["true_label"], predictions["predicted_probability"])
    metrics["skipped_folds"] = int(skipped_folds)
    return {"metrics": metrics, "predictions": predictions, "reader_profiles": profile_frame}


def write_lopo_outputs(result: dict[str, object], out_dir: str | Path) -> None:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result["metrics"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_table(result["predictions"], output / "predictions.csv")
    write_table(result["reader_profiles"], output / "reader_profiles.csv")
