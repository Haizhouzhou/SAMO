from __future__ import annotations

from pathlib import Path

import pandas as pd

from samo_copco.lm_scoring import LMScoringConfig, score_word_table_mock
from samo_copco.profiles import build_reader_profiles
from samo_copco.residualization import crossfit_residualize_by_reader


def test_profiles_include_lm_exposure_and_sensitivity_features() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = pd.read_csv(root / "tests" / "fixtures" / "repeated_participant_word_rows.csv")
    scored = score_word_table_mock(rows, LMScoringConfig(mock_model=True)).frame
    residualized = crossfit_residualize_by_reader(scored)
    profiles = build_reader_profiles(residualized)
    assert len(profiles) == rows["reader_id"].nunique()
    assert "lm_surprisal_exposure_mean" in profiles.columns
    assert "lm_entropy_exposure_mean" in profiles.columns
    assert any(column.startswith("sensitivity__") and column.endswith("__surprisal") for column in profiles.columns)
    assert any(column.startswith("sensitivity__") and column.endswith("__entropy") for column in profiles.columns)
