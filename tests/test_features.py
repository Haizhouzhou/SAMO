from __future__ import annotations

import pandas as pd
import pytest

from samo_copco.features import normalize_feature_columns, select_model_feature_columns, summarize_feature_availability, validate_feature_columns


def test_copco_aliases_normalize_to_public_columns() -> None:
    frame = pd.DataFrame(
        {
            "participantId": ["r1"],
            "speechId": [1],
            "paragraphId": [2],
            "sentenceId": [3],
            "wordId": [4],
            "word": ["alpha"],
            "TRT": [200],
        }
    )
    normalized = normalize_feature_columns(frame)
    assert {"reader_id", "speech_id", "paragraph_id", "sentence_id", "source_word_id", "gaze_duration_ms"} <= set(normalized.columns)


def test_missing_required_gaze_feature_raises_clear_error() -> None:
    frame = pd.DataFrame({"word": ["alpha"]})
    with pytest.raises(ValueError, match="gaze_duration_ms"):
        validate_feature_columns(frame)


def test_blocked_id_label_columns_are_excluded_from_model_predictors() -> None:
    frame = pd.DataFrame(
        {
            "reader_id": ["r1", "r2"],
            "reader_label": [0, 1],
            "lm_surprisal_exposure_mean": [1.0, 2.0],
            "sensitivity__gaze_duration_ms__surprisal": [0.1, 0.2],
        }
    )
    selected = select_model_feature_columns(frame)
    assert "reader_id" not in selected
    assert "reader_label" not in selected
    assert "lm_surprisal_exposure_mean" in selected


def test_feature_summary_reports_lm_families() -> None:
    frame = pd.DataFrame(
        {
            "word": ["alpha"],
            "gaze_duration_ms": [200],
            "lm_word_surprisal": [1.1],
            "lm_word_entropy": [2.2],
            "lm_surprisal_exposure_mean": [1.1],
            "sensitivity__gaze_duration_ms__entropy": [0.2],
        }
    )
    summary = summarize_feature_availability(frame)
    assert "lm_word_surprisal" in summary["predictability_feature_families"]["lm_word_predictability"]
    assert "lm_surprisal_exposure_mean" in summary["residualized_profile_feature_families"]["lm_exposure"]
