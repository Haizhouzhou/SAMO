from __future__ import annotations

from pathlib import Path

import pandas as pd

from samo_copco.data import prepare_word_table
from samo_copco.data_contracts import validate_labels, validate_word_features
from samo_copco.validation import validate_configured_data


def test_word_and_label_contracts_accept_synthetic_fixtures() -> None:
    root = Path(__file__).resolve().parents[1]
    word = pd.read_csv(root / "tests" / "fixtures" / "synthetic_word_features.csv")
    labels = pd.read_csv(root / "tests" / "fixtures" / "synthetic_labels.csv")
    assert validate_word_features(word).ok
    assert validate_labels(labels).ok
    prepared = prepare_word_table(word, labels)
    assert "gaze_log_ms" in prepared.columns
    assert "lm_stable_word_id" in prepared.columns


def test_example_config_dry_run_does_not_open_placeholder_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    message = validate_configured_data(root / "configs" / "copco_paths.example.yaml", dry_run=True)
    assert "dry-run ok" in message
