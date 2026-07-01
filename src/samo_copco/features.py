"""Feature vocabulary, aliases, and predictor selection for SAMO."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .data_contracts import (
    FORBIDDEN_PREDICTOR_COLUMNS,
    GAZE_LOG,
    GAZE_MS,
    LABEL,
    LM_STABLE_WORD_ID,
    PARAGRAPH_ID,
    PREDICTABILITY,
    READER_ID,
    SENTENCE_ID,
    SOURCE_WORD_ID,
    SPEECH_ID,
    TARGET,
    TEXT_ID,
    TRIAL_ID,
    WORD,
    WORD_ID,
    WORD_LENGTH,
    WORD_POSITION,
)

CANONICAL_WORD_COLUMNS = (
    READER_ID,
    TEXT_ID,
    SPEECH_ID,
    PARAGRAPH_ID,
    SENTENCE_ID,
    SOURCE_WORD_ID,
    LM_STABLE_WORD_ID,
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

LM_EXPOSURE_FEATURES = (
    "lm_surprisal_exposure_mean",
    "lm_surprisal_exposure_std",
    "lm_entropy_exposure_mean",
    "lm_entropy_exposure_std",
)

PREDICTABILITY_FEATURES = (
    "lm_word_surprisal",
    "lm_word_entropy",
    PREDICTABILITY,
    WORD_LENGTH,
    WORD_POSITION,
)

RESIDUALIZED_PROFILE_FEATURES = (
    "residual_mean__gaze_duration_ms",
    "residual_std__gaze_duration_ms",
    "sensitivity__gaze_duration_ms__surprisal",
    "sensitivity__gaze_duration_ms__entropy",
    *LM_EXPOSURE_FEATURES,
    "n_words",
)

PREDICTABILITY_FEATURE_FAMILIES = {
    "lm_word_predictability": ("lm_word_surprisal", "lm_word_entropy"),
    "synthetic_predictability": (PREDICTABILITY,),
    "lexical_context": (WORD_LENGTH, WORD_POSITION),
}

RESIDUALIZED_PROFILE_FEATURE_FAMILIES = {
    "lm_exposure": LM_EXPOSURE_FEATURES,
    "residual_location": tuple(column for column in RESIDUALIZED_PROFILE_FEATURES if column.startswith("residual_mean__")),
    "residual_variability": tuple(column for column in RESIDUALIZED_PROFILE_FEATURES if column.startswith("residual_std__")),
    "lm_sensitivity": tuple(column for column in RESIDUALIZED_PROFILE_FEATURES if column.startswith("sensitivity__")),
    "profile_coverage": ("n_words",),
}

COLUMN_ALIASES = {
    "participant": READER_ID,
    "participantid": READER_ID,
    "participant_id": READER_ID,
    "subject": READER_ID,
    "subject_id": READER_ID,
    "subj": READER_ID,
    "reader": READER_ID,
    "reader_id": READER_ID,
    "text": TEXT_ID,
    "text_identifier": TEXT_ID,
    "item_text_id": TEXT_ID,
    "speech": SPEECH_ID,
    "speechid": SPEECH_ID,
    "speech_id": SPEECH_ID,
    "paragraphid": PARAGRAPH_ID,
    "paragraph_id": PARAGRAPH_ID,
    "sentenceid": SENTENCE_ID,
    "sentence_id": SENTENCE_ID,
    "wordid": SOURCE_WORD_ID,
    "source_word_id": SOURCE_WORD_ID,
    "local_word_id": SOURCE_WORD_ID,
    "word_identifier": WORD_ID,
    "global_word_id": WORD_ID,
    "stable_word_id": LM_STABLE_WORD_ID,
    "lm_stable_word_id": LM_STABLE_WORD_ID,
    "trialid": TRIAL_ID,
    "trial_id": TRIAL_ID,
    "orthographic_word": WORD,
    "word_form": WORD,
    "word": WORD,
    "position": WORD_POSITION,
    "word_index": WORD_POSITION,
    "word_position": WORD_POSITION,
    "total_reading_time": GAZE_MS,
    "total_reading_time_ms": GAZE_MS,
    "trt": GAZE_MS,
    "gaze_ms": GAZE_MS,
    "gaze_duration": GAZE_MS,
    "gaze_duration_ms": GAZE_MS,
    "log_gaze_duration_ms": GAZE_LOG,
    "predictability": PREDICTABILITY,
    "probability": PREDICTABILITY,
    "lm_probability": PREDICTABILITY,
    "label": LABEL,
    "class_label": LABEL,
    "group_label": LABEL,
    "reader_label": LABEL,
}

BLOCKED_PREDICTOR_COLUMNS = FORBIDDEN_PREDICTOR_COLUMNS | {
    "participant",
    "subject",
    "reader_group",
    "trial",
    "trial_index",
    "item_id",
    "true_label",
    "predicted_label",
    "predicted_probability",
}

DEFAULT_REQUIRED_COLUMNS = (WORD, GAZE_MS)


def _ordered_present(columns: Iterable[str], candidates: Iterable[str]) -> list[str]:
    available = set(columns)
    return [candidate for candidate in candidates if candidate in available]


def normalize_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with accepted public and CopCo aliases renamed."""

    rename: dict[str, str] = {}
    existing = set(map(str, frame.columns))
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
    """Validate required public feature columns and numeric values."""

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
    """Select numeric public predictors while excluding IDs and targets."""

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
    return {family: _ordered_present(normalized.columns, columns) for family, columns in families.items()}


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
