from pathlib import Path

import pandas as pd
import pytest

from samo_copco.features import (
    infer_available_gaze_features,
    normalize_feature_columns,
    select_model_feature_columns,
    summarize_feature_availability,
    validate_feature_columns,
)


def _synthetic_words() -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    return pd.read_csv(root / "tests" / "fixtures" / "synthetic_word_features.csv")


def test_synthetic_fixture_passes_feature_validation():
    frame = _synthetic_words()
    validate_feature_columns(frame)
    summary = summarize_feature_availability(frame)
    assert "gaze_duration_ms" in summary["available_gaze_features"]
    assert "predictability_score" in summary["predictability_feature_families"]["model_predictability"]


def test_missing_required_gaze_feature_raises_clear_error():
    frame = _synthetic_words().drop(columns=["gaze_duration_ms"])
    with pytest.raises(ValueError, match="missing required feature columns: gaze_duration_ms"):
        validate_feature_columns(frame)


def test_blocked_id_label_text_columns_are_excluded_from_model_predictors():
    frame = _synthetic_words().assign(
        participant_id="synthetic_reader_extra",
        speech_id="synthetic_speech_extra",
        trial_id=1,
        target=0,
        usable_profile_feature=1.25,
    )
    normalized = normalize_feature_columns(frame)
    selected = select_model_feature_columns(normalized)
    blocked = {"reader_id", "participant_id", "reader_label", "target", "speech_id", "text_id", "trial_id"}
    assert not (set(selected) & blocked)
    assert "gaze_duration_ms" in selected
    assert "predictability_score" in selected


def test_feature_family_inference_returns_expected_groups():
    frame = _synthetic_words().assign(
        gaze_residual_mean=0.1,
        predictability_residual_slope=-0.2,
        n_words=24,
    )
    gaze = infer_available_gaze_features(frame)
    summary = summarize_feature_availability(frame)
    assert gaze == ["gaze_duration_ms"]
    assert summary["gaze_feature_families"]["duration"] == ["gaze_duration_ms"]
    assert summary["predictability_feature_families"]["model_predictability"] == ["predictability_score"]
    assert "gaze_residual_mean" in summary["residualized_profile_feature_families"]["residual_location"]
    assert "predictability_residual_slope" in summary["residualized_profile_feature_families"]["residual_sensitivity"]
