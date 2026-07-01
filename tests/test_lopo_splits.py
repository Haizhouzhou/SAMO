from __future__ import annotations

from pathlib import Path

import pandas as pd

from samo_copco.evaluation import assert_reader_disjoint, make_lopo_splits, run_lopo_evaluation
from samo_copco.lm_scoring import LMScoringConfig, score_word_table_mock
from samo_copco.models import select_predictor_columns


def test_lopo_folds_are_reader_disjoint() -> None:
    readers = [f"reader_{idx}" for idx in range(5)]
    splits = make_lopo_splits(readers)
    for split in splits:
        assert_reader_disjoint(split["train_readers"], split["test_readers"])


def test_lopo_evaluation_one_prediction_per_reader() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = pd.read_csv(root / "tests" / "fixtures" / "repeated_participant_word_rows.csv")
    scored = score_word_table_mock(rows, LMScoringConfig(mock_model=True)).frame
    result = run_lopo_evaluation(scored)
    assert len(result["predictions"]) == rows["reader_id"].nunique()
    assert result["metrics"]["n_readers"] == rows["reader_id"].nunique()


def test_identifier_and_target_columns_are_not_predictors() -> None:
    frame = pd.DataFrame({"reader_id": ["r1", "r2"], "reader_label": [0, 1], "target": [0, 1], "usable": [0.1, 0.9]})
    assert select_predictor_columns(frame) == ["usable"]
