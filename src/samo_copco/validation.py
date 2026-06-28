"""Validation routines for public SAMO release inputs and outputs."""

from __future__ import annotations

from pathlib import Path

from .data import load_config, load_configured_inputs, prepare_word_table
from .data_contracts import LABEL, validate_labels, validate_word_features

PLACEHOLDER_PREFIX = "PATH_TO_"


def validate_configured_data(config_path: str | Path, *, dry_run: bool = False) -> str:
    config = load_config(config_path)
    paths = config.get("paths", {})
    word_path = str(paths.get("word_features", ""))
    label_path = str(paths.get("labels", ""))
    if dry_run:
        if word_path.startswith(PLACEHOLDER_PREFIX) or label_path.startswith(PLACEHOLDER_PREFIX):
            return "dry-run ok: real local CopCo paths are user-provided and data are not redistributed"
        return "dry-run ok: config structure parsed; paths were not opened"
    if word_path.startswith(PLACEHOLDER_PREFIX) or label_path.startswith(PLACEHOLDER_PREFIX):
        raise ValueError("replace example paths with local CopCo files before running without dry-run")
    word, labels = load_configured_inputs(config)
    validate_word_features(word, require_label=False).raise_for_errors()
    validate_labels(labels).raise_for_errors()
    prepared = prepare_word_table(word, labels)
    if LABEL not in prepared.columns:
        raise ValueError("prepared table lacks reader labels")
    return f"validated {prepared['reader_id'].nunique()} readers and {len(prepared)} word rows"
