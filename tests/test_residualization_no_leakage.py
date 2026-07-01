from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from samo_copco.lm_scoring import LMScoringConfig, score_word_table_mock
from samo_copco.residualization import crossfit_residualize_by_reader, default_residual_covariates, fit_gaze_residualizers


def _scored_rows() -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    rows = pd.read_csv(root / "tests" / "fixtures" / "repeated_participant_word_rows.csv")
    return score_word_table_mock(rows, LMScoringConfig(mock_model=True)).frame


def test_residualizers_fit_training_readers_only_and_transform_heldout() -> None:
    frame = _scored_rows()
    heldout = {"synthetic_reader_01"}
    train = frame[~frame["reader_id"].isin(heldout)]
    residualizers = fit_gaze_residualizers(train, ["gaze_duration_ms"], default_residual_covariates(frame))
    assert heldout.isdisjoint(residualizers["gaze_duration_ms"].fit_reader_ids_)
    transformed = crossfit_residualize_by_reader(frame)
    assert "resid__gaze_duration_ms" in transformed.columns
    assert transformed["resid__gaze_duration_ms"].notna().all()


def test_residualizer_rejects_identifier_predictors() -> None:
    frame = _scored_rows()
    with pytest.raises(ValueError):
        crossfit_residualize_by_reader(frame, covariates=["reader_id", "lm_word_surprisal"])
