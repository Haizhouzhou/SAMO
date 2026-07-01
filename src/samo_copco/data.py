"""Input loading and word-table preparation for SAMO."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .data_contracts import GAZE_LOG, GAZE_MS, LABEL, READER_ID, WORD, WORD_LENGTH, validate_labels, validate_word_features
from .features import normalize_feature_columns
from .predictability import add_synthetic_proxy_predictability, validate_predictability_columns
from .stimulus_reconstruction import make_lm_stable_word_id, normalize_copco_columns


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("config must contain a mapping at top level")
    return payload


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"table not found: {table_path}")
    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix == ".tsv":
        return pd.read_csv(table_path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    raise ValueError(f"unsupported table extension: {suffix}")


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(output_path, index=False)
        return
    if suffix == ".tsv":
        frame.to_csv(output_path, sep="\t", index=False)
        return
    raise ValueError("public writer supports .csv and .tsv outputs")


def normalize_columns(frame: pd.DataFrame, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    out = normalize_copco_columns(frame) if WORD in set(frame.columns) or "word_form" in set(frame.columns) else normalize_feature_columns(frame)
    if mapping:
        reverse = {source: public for public, source in mapping.items() if source in out.columns}
        out = out.rename(columns=reverse).copy()
    return out


def prepare_word_table(
    word_features: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    *,
    allow_synthetic_proxy: bool = False,
) -> pd.DataFrame:
    """Normalize, validate, and enrich repeated word rows."""

    out = make_lm_stable_word_id(word_features)
    if labels is not None and LABEL not in out.columns:
        labels_norm = normalize_copco_columns(labels)
        validate_labels(labels_norm).raise_for_errors()
        out = out.merge(labels_norm[[READER_ID, LABEL]], on=READER_ID, how="left", validate="many_to_one")
    if WORD_LENGTH not in out.columns:
        out[WORD_LENGTH] = out[WORD].astype(str).str.len().astype(float)
    validate_word_features(out, require_label=LABEL in out.columns).raise_for_errors()
    if {"lm_word_surprisal", "lm_word_entropy"} <= set(out.columns):
        validate_predictability_columns(out)
    elif "predictability_score" in out.columns:
        validate_predictability_columns(out, allow_synthetic_score=True)
    elif allow_synthetic_proxy:
        out = add_synthetic_proxy_predictability(out)
    gaze = pd.to_numeric(out[GAZE_MS], errors="coerce")
    if gaze.isna().any():
        raise ValueError(f"{GAZE_MS} contains non-numeric values")
    out[GAZE_LOG] = gaze.map(lambda value: math.log(float(value)))
    if LABEL in out.columns:
        out[LABEL] = pd.to_numeric(out[LABEL], errors="raise").astype(int)
    return out


def load_configured_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = config.get("paths", {})
    columns = config.get("columns", {})
    word_path = paths.get("word_features")
    label_path = paths.get("labels")
    if not word_path or not label_path:
        raise ValueError("config paths.word_features and paths.labels are required")
    word = normalize_columns(read_table(word_path), columns)
    labels = normalize_columns(read_table(label_path), columns)
    return word, labels


def synthetic_fixture_paths(repo_root: Path) -> tuple[Path, Path]:
    fixture_dir = repo_root / "tests" / "fixtures"
    return fixture_dir / "repeated_participant_word_rows.csv", fixture_dir / "synthetic_labels.csv"
