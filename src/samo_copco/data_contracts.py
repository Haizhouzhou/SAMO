"""Public data contracts for SAMO tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

READER_ID = "reader_id"
TEXT_ID = "text_id"
SPEECH_ID = "speech_id"
PARAGRAPH_ID = "paragraph_id"
SENTENCE_ID = "sentence_id"
SOURCE_WORD_ID = "source_word_id"
WORD_ID = "word_id"
LM_STABLE_WORD_ID = "lm_stable_word_id"
TRIAL_ID = "trial_id"
WORD = "word"
GAZE_MS = "gaze_duration_ms"
GAZE_LOG = "gaze_log_ms"
LABEL = "reader_label"
TARGET = "target"
PREDICTABILITY = "predictability_score"
WORD_LENGTH = "word_length"
WORD_POSITION = "word_position"

REQUIRED_LABEL_COLUMNS = (READER_ID, LABEL)
MINIMAL_WORD_COLUMNS = (WORD, GAZE_MS)
LM_REQUIRED_OUTPUT_COLUMNS = (
    LM_STABLE_WORD_ID,
    WORD,
    "lm_model_id",
    "lm_tokenizer_id",
    "lm_context_mode",
    "lm_word_surprisal",
    "lm_word_entropy",
    "lm_word_entropy_onset",
    "lm_subword_count",
    "lm_scored_subword_count",
    "lm_alignment_status",
)

IDENTIFIER_COLUMNS = {
    READER_ID,
    "participant",
    "participant_id",
    "participantId",
    "subject_id",
    "reader",
    "direct_reader_id",
    TEXT_ID,
    SPEECH_ID,
    PARAGRAPH_ID,
    SENTENCE_ID,
    SOURCE_WORD_ID,
    WORD_ID,
    LM_STABLE_WORD_ID,
    TRIAL_ID,
}
TARGET_COLUMNS = {LABEL, TARGET, "label", "class_label", "dyslexia_label", "dyslexia_labelled", "y", "y_true"}
FORBIDDEN_PREDICTOR_COLUMNS = IDENTIFIER_COLUMNS | TARGET_COLUMNS


@dataclass(frozen=True)
class ValidationResult:
    """Machine-readable validation result."""

    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.errors))


def missing_columns(frame: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return [column for column in required if column not in frame.columns]


def validate_labels(frame: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    missing = missing_columns(frame, REQUIRED_LABEL_COLUMNS)
    if missing:
        errors.append("missing columns: " + ", ".join(missing))
    if not errors:
        if frame[READER_ID].isna().any():
            errors.append(f"null values in {READER_ID}")
        duplicate_count = int(frame[READER_ID].duplicated().sum())
        if duplicate_count:
            errors.append(f"duplicate labels per reader: {duplicate_count}")
        labels = pd.to_numeric(frame[LABEL], errors="coerce")
        if labels.isna().any():
            errors.append(f"non-numeric values in {LABEL}")
        unique = set(labels.dropna().astype(int).tolist())
        if not unique <= {0, 1}:
            errors.append(f"{LABEL} must be binary 0 or 1")
    return ValidationResult(ok=not errors, errors=tuple(errors))


def validate_word_features(frame: pd.DataFrame, *, require_label: bool = False) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    missing = missing_columns(frame, MINIMAL_WORD_COLUMNS)
    if missing:
        errors.append("missing columns: " + ", ".join(missing))
    if require_label and LABEL not in frame.columns:
        errors.append("missing columns: " + LABEL)
    if not errors:
        if frame[WORD].isna().any():
            errors.append(f"null values in {WORD}")
        gaze = pd.to_numeric(frame[GAZE_MS], errors="coerce")
        if gaze.isna().any():
            errors.append(f"non-numeric values in {GAZE_MS}")
        if (gaze <= 0).any():
            errors.append(f"non-positive values in {GAZE_MS}")
        if READER_ID in frame.columns and frame[READER_ID].isna().any():
            errors.append(f"null values in {READER_ID}")
        if require_label:
            label_result = validate_labels(frame[[READER_ID, LABEL]].drop_duplicates())
            errors.extend(label_result.errors)
        if LM_STABLE_WORD_ID not in frame.columns:
            warnings.append(f"{LM_STABLE_WORD_ID} has not been constructed yet")
    return ValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def assert_no_forbidden_predictors(columns: Iterable[str]) -> None:
    forbidden = sorted(set(columns) & FORBIDDEN_PREDICTOR_COLUMNS)
    if forbidden:
        raise ValueError("forbidden predictor columns: " + ", ".join(forbidden))
