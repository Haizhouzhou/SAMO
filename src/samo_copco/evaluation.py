"""Reader-disjoint LOPO evaluation for SAMO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data import prepare_word_table, write_table
from .data_contracts import LABEL, READER_ID
from .metrics import classification_metrics
from .models import RidgeLinearClassifier, select_predictor_columns
from .profiles import build_reader_profiles
from .residualization import fit_transform_fold_local


def make_lopo_splits(readers: Iterable[str]) -> list[dict[str, object]]:
    reader_list = sorted({str(reader) for reader in readers})
    if len(reader_list) < 3:
        raise ValueError("LOPO evaluation requires at least three readers")
    splits: list[dict[str, object]] = []
    for fold_id, heldout in enumerate(reader_list, start=1):
        train = [reader for reader in reader_list if reader != heldout]
        splits.append({"fold_id": fold_id, "train_readers": train, "test_readers": [heldout], "heldout_reader": heldout})
    return splits


def assert_reader_disjoint(train_readers: Iterable[str], test_readers: Iterable[str]) -> None:
    overlap = set(map(str, train_readers)) & set(map(str, test_readers))
    if overlap:
        raise ValueError("reader overlap between train and test: " + ", ".join(sorted(overlap)))


def run_lopo_evaluation(
    word_features: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    *,
    model_columns: Iterable[str] | None = None,
    alpha: float = 1.0,
) -> dict[str, pd.DataFrame | dict[str, float | int]]:
    prepared = prepare_word_table(word_features, labels)
    if LABEL not in prepared.columns:
        raise ValueError("reader labels are required for LOPO evaluation")
    readers = sorted(prepared[READER_ID].astype(str).unique())
    splits = make_lopo_splits(readers)
    prediction_rows: list[dict[str, object]] = []
    heldout_profiles: list[pd.DataFrame] = []
    for split in splits:
        train_readers = set(split["train_readers"])
        test_readers = set(split["test_readers"])
        assert_reader_disjoint(train_readers, test_readers)
        train_words = prepared[prepared[READER_ID].astype(str).isin(train_readers)].copy()
        fold_rows = prepared[prepared[READER_ID].astype(str).isin(train_readers | test_readers)].copy()
        transformed, residualizer = fit_transform_fold_local(
            train_words,
            fold_rows,
            heldout_readers=test_readers,
        )
        if set(residualizer.fit_reader_ids_) & test_readers:
            raise RuntimeError("residualizer fit readers overlap held-out readers")
        profiles = build_reader_profiles(transformed)
        train_profiles = profiles[profiles[READER_ID].astype(str).isin(train_readers)].copy()
        test_profiles = profiles[profiles[READER_ID].astype(str).isin(test_readers)].copy()
        columns = select_predictor_columns(train_profiles, model_columns)
        model = RidgeLinearClassifier(alpha=alpha).fit(train_profiles, columns=columns)
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
    predictions = pd.DataFrame(prediction_rows).sort_values(["fold_id", READER_ID]).reset_index(drop=True)
    profile_frame = pd.concat(heldout_profiles, ignore_index=True).sort_values(READER_ID).reset_index(drop=True)
    if profile_frame[READER_ID].duplicated().any():
        raise RuntimeError("cross-fitted profile output must have one row per reader")
    metrics = classification_metrics(predictions["true_label"], predictions["predicted_probability"])
    return {"metrics": metrics, "predictions": predictions, "reader_profiles": profile_frame}


def write_lopo_outputs(result: dict[str, object], out_dir: str | Path) -> None:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_table(result["predictions"], output / "predictions.csv")
    write_table(result["reader_profiles"], output / "reader_profiles.csv")
