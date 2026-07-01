from __future__ import annotations

from pathlib import Path

import pandas as pd

from samo_copco.lm_scoring import LMScoringConfig, score_word_table_mock
from samo_copco.predictability import merge_lm_features


def test_repeated_participant_word_rows_merge_many_to_one_by_stable_id() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = pd.read_csv(root / "tests" / "fixtures" / "repeated_participant_word_rows.csv")
    result = score_word_table_mock(rows, LMScoringConfig(mock_model=True))
    lm = result.lm_features
    assert lm["lm_stable_word_id"].is_unique
    merged = merge_lm_features(rows, lm)
    assert len(merged) == len(rows)
    assert merged["reader_id"].nunique() == 8
    assert merged["lm_word_surprisal"].notna().all()
    assert merged.groupby("lm_stable_word_id")["lm_word_surprisal"].nunique().max() == 1
