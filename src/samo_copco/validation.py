"""Validation routines for public SAMO inputs and outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import load_config, load_configured_inputs, prepare_word_table
from .data_contracts import LABEL, LM_REQUIRED_OUTPUT_COLUMNS, LM_STABLE_WORD_ID, READER_ID, ValidationResult, validate_labels, validate_word_features
from .features import normalize_feature_columns
from .lm_scoring import validate_lm_feature_table
from .models import select_predictor_columns
from .stimulus_reconstruction import make_lm_stable_word_id, normalize_copco_columns

PLACEHOLDER_PREFIX = "PATH_TO_"


def validate_configured_data(config_path: str | Path, *, dry_run: bool = False) -> str:
    config = load_config(config_path)
    paths = config.get("paths", {})
    word_path = str(paths.get("word_features", ""))
    label_path = str(paths.get("labels", ""))
    if dry_run:
        if word_path.startswith(PLACEHOLDER_PREFIX) or label_path.startswith(PLACEHOLDER_PREFIX):
            return "dry-run ok: replace example paths with local CopCo tables before real runs"
        return "dry-run ok: config parsed; paths were not opened"
    if word_path.startswith(PLACEHOLDER_PREFIX) or label_path.startswith(PLACEHOLDER_PREFIX):
        raise ValueError("replace example paths before running without dry-run")
    word, labels = load_configured_inputs(config)
    word_norm = make_lm_stable_word_id(word)
    validate_word_features(word_norm, require_label=False).raise_for_errors()
    validate_labels(normalize_feature_columns(labels)).raise_for_errors()
    prepared = prepare_word_table(word, labels, allow_synthetic_proxy=False)
    if LABEL not in prepared.columns:
        raise ValueError("prepared table lacks reader labels")
    return f"validated {prepared[READER_ID].nunique()} readers and {len(prepared)} word rows"


def validate_lm_output_columns(frame: pd.DataFrame) -> ValidationResult:
    errors = []
    missing = [column for column in LM_REQUIRED_OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        errors.append("missing LM output columns: " + ", ".join(missing))
    if not errors:
        try:
            validate_lm_feature_table(frame).raise_for_errors()
        except ValueError as exc:
            errors.append(str(exc))
    return ValidationResult(ok=not errors, errors=tuple(errors))


def validate_profile_table(frame: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    required = [READER_ID, "lm_surprisal_exposure_mean", "lm_entropy_exposure_mean"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        errors.append("missing profile columns: " + ", ".join(missing))
    if READER_ID in frame.columns and frame[READER_ID].duplicated().any():
        errors.append("profile table must have one row per reader")
    if not any(column.startswith("sensitivity__") and column.endswith("__surprisal") for column in frame.columns):
        errors.append("missing surprisal sensitivity columns")
    if not any(column.startswith("sensitivity__") and column.endswith("__entropy") for column in frame.columns):
        errors.append("missing entropy sensitivity columns")
    return ValidationResult(ok=not errors, errors=tuple(errors))


def validate_many_to_one_lm_merge(word_rows: pd.DataFrame, lm_features: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    words = make_lm_stable_word_id(word_rows)
    lm = make_lm_stable_word_id(lm_features)
    if lm[LM_STABLE_WORD_ID].duplicated().any():
        errors.append("LM feature rows duplicate lm_stable_word_id")
    missing = sorted(set(words[LM_STABLE_WORD_ID]) - set(lm[LM_STABLE_WORD_ID]))
    if missing:
        errors.append(f"LM features missing {len(missing)} stable word ids")
    return ValidationResult(ok=not errors, errors=tuple(errors))


def assert_no_forbidden_predictor_columns(frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    select_predictor_columns(frame, columns)
