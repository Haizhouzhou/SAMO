"""Public data contracts for SAMO inputs and intermediate tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

READER_ID = "reader_id"
TEXT_ID = "text_id"
SPEECH_ID = "speech_id"
WORD_ID = "word_id"
WORD = "word"
GAZE_MS = "gaze_duration_ms"
LABEL = "reader_label"
TARGET = "target"
PREDICTABILITY = "predictability_score"
WORD_LENGTH = "word_length"
WORD_POSITION = "word_position"
GAZE_LOG = "gaze_log_ms"

REQUIRED_WORD_COLUMNS = (READER_ID, TEXT_ID, WORD_ID, WORD, GAZE_MS)
REQUIRED_LABEL_COLUMNS = (READER_ID, LABEL)

IDENTIFIER_COLUMNS = {
    READER_ID,
    "participant_id",
    "reader",
    "direct_reader_id",
    TEXT_ID,
    SPEECH_ID,
    WORD_ID,
    "sentence_id",
    "paragraph_id",
}
TARGET_COLUMNS = {LABEL, TARGET, "label", "dyslexia_label", "dyslexia_labelled", "y", "y_true"}
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


def validate_word_features(frame: pd.DataFrame, *, require_label: bool = False) -> ValidationResult:
    errors: list[str] = []
    required = list(REQUIRED_WORD_COLUMNS)
    if require_label:
        required.append(LABEL)
    missing = missing_columns(frame, required)
    if missing:
        errors.append("missing columns: " + ", ".join(missing))
    if not errors:
        key_columns = [READER_ID, TEXT_ID, WORD_ID]
        duplicate_count = int(frame.duplicated(key_columns).sum())
        if duplicate_count:
            errors.append(f"duplicate reader-text-word rows: {duplicate_count}")
        for column in [READER_ID, TEXT_ID, WORD_ID, WORD]:
            if frame[column].isna().any():
                errors.append(f"null values in {column}")
        gaze = pd.to_numeric(frame[GAZE_MS], errors="coerce")
        if gaze.isna().any():
            errors.append(f"non-numeric values in {GAZE_MS}")
        if (gaze <= 0).any():
            errors.append(f"non-positive values in {GAZE_MS}")
        if require_label:
            validate_labels(frame[[READER_ID, LABEL]].drop_duplicates()).errors and errors.extend(
                validate_labels(frame[[READER_ID, LABEL]].drop_duplicates()).errors
            )
    return ValidationResult(ok=not errors, errors=tuple(errors))


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


def assert_no_forbidden_predictors(columns: Iterable[str]) -> None:
    forbidden = sorted(set(columns) & FORBIDDEN_PREDICTOR_COLUMNS)
    if forbidden:
        raise ValueError("forbidden predictor columns: " + ", ".join(forbidden))
