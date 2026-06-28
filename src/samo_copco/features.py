"""Feature vocabulary and column selection helpers for SAMO."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .data_contracts import (
    FORBIDDEN_PREDICTOR_COLUMNS,
    GAZE_LOG,
    GAZE_MS,
    LABEL,
    PREDICTABILITY,
    READER_ID,
    SPEECH_ID,
    TARGET,
    TEXT_ID,
    WORD,
    WORD_ID,
    WORD_LENGTH,
    WORD_POSITION,
)

CANONICAL_WORD_COLUMNS = (
    READER_ID,
    TEXT_ID,
    WORD_ID,
    WORD,
    WORD_POSITION,
)

CANONICAL_GAZE_FEATURES = (
    GAZE_MS,
    GAZE_LOG,
    "first_fixation_ms",
    "single_fixation_ms",
    "go_past_ms",
    "n_fixations",
    "rereading_ms",
    "skip_rate",
)

GAZE_FEATURE_FAMILIES = {
    "duration": (GAZE_MS, GAZE_LOG, "first_fixation_ms", "single_fixation_ms", "go_past_ms", "rereading_ms"),
    "count": ("n_fixations",),
    "binary_or_rate": ("skip_rate",),
}

PREDICTABILITY_FEATURES = (
    PREDICTABILITY,
    "surprisal",
    "negative_log_probability",
    "word_frequency_log",
    WORD_LENGTH,
    WORD_POSITION,
)

PREDICTABILITY_FEATURE_FAMILIES = {
    "model_predictability": (PREDICTABILITY, "surprisal", "negative_log_probability"),
    "lexical_context": ("word_frequency_log", WORD_LENGTH, WORD_POSITION),
}

RESIDUALIZED_PROFILE_FEATURES = (
    "gaze_residual",
    "gaze_residual_mean",
    "gaze_residual_std",
    "predictability_residual_slope",
    "length_residual_slope",
    "gaze_ms_mean",
    "predictability_mean",
    "predictability_std",
    "n_words",
)

RESIDUALIZED_PROFILE_FEATURE_FAMILIES = {
    "residual_location": ("gaze_residual", "gaze_residual_mean"),
    "residual_variability": ("gaze_residual_std",),
    "residual_sensitivity": ("predictability_residual_slope", "length_residual_slope"),
    "profile_coverage": ("n_words",),
    "profile_means": ("gaze_ms_mean", "predictability_mean", "predictability_std"),
}

COLUMN_ALIASES = {
    "participant": READER_ID,
    "participant_id": READER_ID,
    "subject": READER_ID,
    "subject_id": READER_ID,
    "subj": READER_ID,
    "reader": READER_ID,
    "text": TEXT_ID,
    "text_identifier": TEXT_ID,
    "item_text_id": TEXT_ID,
    "speech": SPEECH_ID,
    "speech_identifier": SPEECH_ID,
    "word_identifier": WORD_ID,
    "orthographic_word": WORD,
    "word_form": WORD,
    "position": WORD_POSITION,
    "word_index": WORD_POSITION,
    "total_reading_time": GAZE_MS,
    "total_reading_time_ms": GAZE_MS,
    "trt": GAZE_MS,
    "gaze_ms": GAZE_MS,
    "gaze_duration": GAZE_MS,
    "log_gaze_duration_ms": GAZE_LOG,
    "predictability": PREDICTABILITY,
    "probability": PREDICTABILITY,
    "lm_probability": PREDICTABILITY,
    "label": LABEL,
    "class_label": LABEL,
    "group_label": LABEL,
}

BLOCKED_PREDICTOR_COLUMNS = FORBIDDEN_PREDICTOR_COLUMNS | {
    "participant",
    "participant_id",
    "subject",
    "subject_id",
    "subj",
    "reader",
    "reader_group",
    "direct_reader_id",
    "text",
    "text_identifier",
    "item_text_id",
    "trial",
    "trial_id",
    "trial_index",
    "item_id",
    "sentence_id",
    "paragraph_id",
    "class_label",
    "group_label",
    "true_label",
    "predicted_label",
    "predicted_probability",
    SPEECH_ID,
    TEXT_ID,
    WORD_ID,
    LABEL,
    TARGET,
}

DEFAULT_REQUIRED_COLUMNS = (READER_ID, TEXT_ID, WORD_ID, WORD, GAZE_MS)


def _ordered_present(columns: Iterable[str], candidates: Iterable[str]) -> list[str]:
    available = set(columns)
    return [candidate for candidate in candidates if candidate in available]


def normalize_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with accepted public aliases renamed to canonical names."""

    rename: dict[str, str] = {}
    existing = set(frame.columns)
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for alias, canonical in COLUMN_ALIASES.items():
        source = lowered.get(alias.lower())
        if source is None or source == canonical:
            continue
        if canonical in existing:
            continue
        rename[source] = canonical
        existing.add(canonical)
    return frame.rename(columns=rename).copy()


def validate_feature_columns(frame: pd.DataFrame, required: Iterable[str] | None = None) -> None:
    """Validate that required public feature columns exist and are usable."""

    normalized = normalize_feature_columns(frame)
    required_columns = tuple(required) if required is not None else DEFAULT_REQUIRED_COLUMNS
    missing = [column for column in required_columns if column not in normalized.columns]
    if missing:
        raise ValueError("missing required feature columns: " + ", ".join(missing))
    numeric_required = [column for column in required_columns if column in CANONICAL_GAZE_FEATURES]
    numeric_required.extend(column for column in (PREDICTABILITY, WORD_LENGTH, WORD_POSITION) if column in required_columns)
    for column in dict.fromkeys(numeric_required):
        values = pd.to_numeric(normalized[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"feature column {column} must be numeric")
        if column in {GAZE_MS, "first_fixation_ms", "single_fixation_ms", "go_past_ms", "rereading_ms"} and (values <= 0).any():
            raise ValueError(f"feature column {column} must be positive")


def infer_available_gaze_features(frame: pd.DataFrame) -> list[str]:
    """Return canonical gaze features present in a table, preserving public order."""

    normalized = normalize_feature_columns(frame)
    return _ordered_present(normalized.columns, CANONICAL_GAZE_FEATURES)


def _feature_candidates(include_predictability: bool, include_residual: bool) -> tuple[str, ...]:
    candidates: list[str] = list(CANONICAL_GAZE_FEATURES)
    if include_predictability:
        candidates.extend(PREDICTABILITY_FEATURES)
    if include_residual:
        candidates.extend(RESIDUALIZED_PROFILE_FEATURES)
    return tuple(dict.fromkeys(candidates))


def select_model_feature_columns(
    frame: pd.DataFrame,
    *,
    include_predictability: bool = True,
    include_residual: bool = True,
) -> list[str]:
    """Select numeric public predictor columns while excluding IDs and targets."""

    normalized = normalize_feature_columns(frame)
    blocked = set(BLOCKED_PREDICTOR_COLUMNS)
    selected: list[str] = []
    for column in _feature_candidates(include_predictability, include_residual):
        if column in blocked or column not in normalized.columns:
            continue
        if pd.api.types.is_numeric_dtype(normalized[column]):
            selected.append(column)
    if not selected:
        raise ValueError("no usable public model feature columns available")
    return selected


def _available_by_family(frame: pd.DataFrame, families: dict[str, tuple[str, ...]]) -> dict[str, list[str]]:
    normalized = normalize_feature_columns(frame)
    return {
        family: _ordered_present(normalized.columns, columns)
        for family, columns in families.items()
    }


def summarize_feature_availability(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize available public feature families and predictor eligibility."""

    normalized = normalize_feature_columns(frame)
    missing_required = [column for column in DEFAULT_REQUIRED_COLUMNS if column not in normalized.columns]
    try:
        selected = select_model_feature_columns(normalized)
    except ValueError:
        selected = []
    return {
        "available_gaze_features": infer_available_gaze_features(normalized),
        "gaze_feature_families": _available_by_family(normalized, GAZE_FEATURE_FAMILIES),
        "predictability_feature_families": _available_by_family(normalized, PREDICTABILITY_FEATURE_FAMILIES),
        "residualized_profile_feature_families": _available_by_family(normalized, RESIDUALIZED_PROFILE_FEATURE_FAMILIES),
        "blocked_predictor_columns_present": sorted(set(normalized.columns) & BLOCKED_PREDICTOR_COLUMNS),
        "selected_model_feature_columns": selected,
        "missing_required_columns": missing_required,
    }
