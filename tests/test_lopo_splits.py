from pathlib import Path

import pandas as pd

from samo_copco.data import prepare_word_table
from samo_copco.evaluation import assert_reader_disjoint, make_lopo_splits, run_lopo_evaluation
from samo_copco.models import select_predictor_columns


def _fixtures():
    root = Path(__file__).resolve().parents[1]
    word = pd.read_csv(root / "tests" / "fixtures" / "synthetic_word_features.csv")
    labels = pd.read_csv(root / "tests" / "fixtures" / "synthetic_labels.csv")
    return word, labels


def test_lopo_splits_are_reader_disjoint():
    word, labels = _fixtures()
    prepared = prepare_word_table(word, labels)
    splits = make_lopo_splits(prepared["reader_id"])
    assert len(splits) == prepared["reader_id"].nunique()
    for split in splits:
        assert_reader_disjoint(split["train_readers"], split["test_readers"])


def test_identifier_and_target_columns_are_not_predictors():
    frame = pd.DataFrame(
        {
            "reader_id": ["synthetic_reader_01", "synthetic_reader_02"],
            "text_id": ["synthetic_text_01", "synthetic_text_02"],
            "speech_id": ["synthetic_speech_01", "synthetic_speech_02"],
            "reader_label": [0, 1],
            "target": [0, 1],
            "usable_feature": [0.1, 0.9],
        }
    )
    assert select_predictor_columns(frame) == ["usable_feature"]


def test_lopo_evaluation_writes_one_prediction_per_reader():
    word, labels = _fixtures()
    result = run_lopo_evaluation(word, labels)
    predictions = result["predictions"]
    profiles = result["reader_profiles"]
    assert len(predictions) == labels["reader_id"].nunique()
    assert len(profiles) == labels["reader_id"].nunique()
    assert result["metrics"]["n_readers"] == labels["reader_id"].nunique()
