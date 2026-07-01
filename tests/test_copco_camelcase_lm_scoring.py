from __future__ import annotations

from pathlib import Path

import pandas as pd

from samo_copco.lm_scoring import LMScoringConfig, score_word_table_mock


def test_copco_camelcase_lm_scoring_preserves_local_wordid_reuse() -> None:
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(root / "tests" / "fixtures" / "copco_camelcase_lm_words.csv")
    result = score_word_table_mock(frame, LMScoringConfig(mock_model=True))
    out = result.frame
    assert {"lm_stable_word_id", "speech_id", "paragraph_id", "sentence_id", "source_word_id"} <= set(out.columns)
    assert out["lm_stable_word_id"].is_unique
    assert out["source_word_id"].astype(str).tolist().count("0") == 2
    assert out["lm_stable_word_id"].nunique() == len(frame)
