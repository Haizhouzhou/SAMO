"""Causal-LM surprisal and entropy scoring for SAMO."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import write_table
from .data_contracts import LM_REQUIRED_OUTPUT_COLUMNS, LM_STABLE_WORD_ID
from .stimulus_reconstruction import StimulusContext, group_word_rows_for_lm, prepare_stimulus_word_table
from .token_alignment import (
    TokenSpan,
    aggregate_token_scores_to_words,
    align_token_offsets_to_word_spans,
    validate_token_word_alignment,
)

DEFAULT_DFM_MODEL_ID = "danish-foundation-models/dfm-decoder-open-v0-7b-pt"
MOCK_MODEL_ID = "deterministic-local-mock-causal-lm"


@dataclass(frozen=True)
class LMScoringConfig:
    """Configuration for word-level causal-LM scoring."""

    model_id: str = DEFAULT_DFM_MODEL_ID
    tokenizer_id: str | None = None
    context_mode: str = "paragraph"
    device: str = "auto"
    dtype: str = "auto"
    mock_model: bool = False
    dry_run: bool = False
    shard_id: int = 0
    num_shards: int = 1
    log_base: str = "natural"


@dataclass(frozen=True)
class LMOutputValidationResult:
    ok: bool
    errors: tuple[str, ...]

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.errors))


@dataclass(frozen=True)
class LMScoringResult:
    """Scored output and sidecar payloads."""

    frame: pd.DataFrame
    lm_features: pd.DataFrame
    manifest: dict[str, Any]
    alignment_report: dict[str, Any]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def causal_next_token_scores(logits: Any, input_ids: Any) -> tuple[np.ndarray, np.ndarray]:
    """Compute shifted next-token surprisal and entropy in natural log units."""

    logits_array = np.asarray(logits, dtype=np.float64)
    ids = np.asarray(input_ids, dtype=np.int64)
    if logits_array.ndim == 3:
        if logits_array.shape[0] != 1:
            raise ValueError("one sequence expected")
        logits_array = logits_array[0]
    if ids.ndim == 2:
        if ids.shape[0] != 1:
            raise ValueError("one sequence expected")
        ids = ids[0]
    if logits_array.ndim != 2 or ids.ndim != 1:
        raise ValueError("logits must be sequence x vocab and input_ids must be sequence")
    if logits_array.shape[0] != ids.shape[0]:
        raise ValueError("logits and input_ids lengths differ")
    if len(ids) < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    shifted = logits_array[:-1]
    observed_next = ids[1:]
    max_logits = shifted.max(axis=-1, keepdims=True)
    stable = shifted - max_logits
    log_norm = max_logits + np.log(np.exp(stable).sum(axis=-1, keepdims=True))
    log_probs = shifted - log_norm
    probs = np.exp(log_probs)
    surprisal = -log_probs[np.arange(len(observed_next)), observed_next]
    entropy = -(probs * log_probs).sum(axis=-1)
    return surprisal.astype(float), entropy.astype(float)


def _torch_dtype(torch_module: Any, dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    mapping = {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError("dtype must be auto, float32, float16, or bfloat16")
    return mapping[dtype]


def load_causal_lm(
    model_id: str,
    tokenizer_id: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
) -> tuple[Any, Any, Any]:
    """Load a Hugging Face causal LM and fast tokenizer."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("real LM scoring requires installing the optional lm dependencies") from exc

    tokenizer_name = tokenizer_id or model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(f"fast tokenizer with offset mapping is required: {tokenizer_name}")
    kwargs: dict[str, Any] = {"torch_dtype": _torch_dtype(torch, dtype)}
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device in {"cpu", "cuda"}:
        model = model.to(device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model.eval()
    return tokenizer, model, torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")


def _score_rows_from_token_data(
    context: StimulusContext,
    token_offsets: list[tuple[int, int]],
    token_scores: dict[int, dict[str, float]],
    *,
    model_id: str,
    tokenizer_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    token_spans = [
        TokenSpan(index, int(start), int(end), is_special=int(end) <= int(start))
        for index, (start, end) in enumerate(token_offsets)
    ]
    assignments = align_token_offsets_to_word_spans(token_spans, context.word_spans)
    report = validate_token_word_alignment(assignments, context.word_spans)
    words = aggregate_token_scores_to_words(assignments, token_scores, context.word_spans)
    words["lm_model_id"] = model_id
    words["lm_tokenizer_id"] = tokenizer_id
    words["lm_context_mode"] = context.context_mode
    words["lm_context_id"] = context.context_id
    words["lm_context_tokens"] = int(len(token_offsets))
    report["context_id"] = context.context_id
    return words, report


def score_context_with_transformers(
    context_text: str,
    tokenizer: Any,
    model: Any,
    device: Any,
    *,
    word_spans: tuple[Any, ...],
    context_id: str,
    context_mode: str,
    model_id: str,
    tokenizer_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score one reconstructed context with a loaded causal LM."""

    encoded = tokenizer(context_text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=True)
    offsets = [(int(start), int(end)) for start, end in encoded.pop("offset_mapping")[0].tolist()]
    encoded = {key: value.to(device) for key, value in encoded.items()}
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for real LM scoring") from exc
    with torch.inference_mode():
        output = model(**encoded)
        logits = output.logits[0].float().detach().cpu().numpy()
        input_ids = encoded["input_ids"][0].detach().cpu().numpy()
    surprisal, entropy = causal_next_token_scores(logits, input_ids)
    token_scores = {
        token_index: {"surprisal": float(surprisal[token_index - 1]), "entropy": float(entropy[token_index - 1])}
        for token_index in range(1, len(input_ids))
    }
    context = StimulusContext(context_id, context_mode, context_text, tuple(word_spans))
    return _score_rows_from_token_data(context, offsets, token_scores, model_id=model_id, tokenizer_id=tokenizer_id)


def _mock_offsets_for_context(context: StimulusContext) -> list[tuple[int, int]]:
    offsets = [(0, 0)]
    for span in context.word_spans:
        width = span.end - span.start
        if width > 6:
            midpoint = span.start + max(1, width // 2)
            offsets.append((span.start, midpoint))
            offsets.append((midpoint, span.end))
        else:
            offsets.append((span.start, span.end))
    return offsets


def score_context_with_mock_model(
    context_text: str,
    word_spans: tuple[Any, ...],
    *,
    context_id: str = "mock_context",
    context_mode: str = "paragraph",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score one context with deterministic local token scores."""

    context = StimulusContext(context_id, context_mode, context_text, tuple(word_spans))
    offsets = _mock_offsets_for_context(context)
    token_scores: dict[int, dict[str, float]] = {}
    for token_index, (start, end) in enumerate(offsets[1:], start=1):
        width = max(1, int(end) - int(start))
        token_scores[token_index] = {
            "surprisal": float(math.log1p(width) + 0.03 * token_index),
            "entropy": float(math.log(2.0 + (token_index % 7)) + 0.01 * min(width, 10)),
        }
    return _score_rows_from_token_data(
        context,
        offsets,
        token_scores,
        model_id=MOCK_MODEL_ID,
        tokenizer_id=MOCK_MODEL_ID,
    )


def _merge_if_repeated_input(input_frame: pd.DataFrame, lm_features: pd.DataFrame) -> pd.DataFrame:
    normalized = prepare_stimulus_word_table(input_frame)
    if len(normalized) == len(input_frame):
        return lm_features.copy()
    from .predictability import merge_lm_features

    return merge_lm_features(input_frame, lm_features)


def _score_contexts(contexts: list[StimulusContext], config: LMScoringConfig) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    assigned = contexts[config.shard_id :: config.num_shards]
    reports: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    if config.mock_model:
        for context in assigned:
            frame, report = score_context_with_mock_model(
                context.context_text,
                context.word_spans,
                context_id=context.context_id,
                context_mode=context.context_mode,
            )
            frames.append(frame)
            reports.append(report)
    else:
        tokenizer, model, device = load_causal_lm(config.model_id, config.tokenizer_id, config.device, config.dtype)
        tokenizer_name = str(getattr(tokenizer, "name_or_path", config.tokenizer_id or config.model_id))
        for context in assigned:
            frame, report = score_context_with_transformers(
                context.context_text,
                tokenizer,
                model,
                device,
                word_spans=context.word_spans,
                context_id=context.context_id,
                context_mode=context.context_mode,
                model_id=config.model_id,
                tokenizer_id=tokenizer_name,
            )
            frames.append(frame)
            reports.append(report)
    if frames:
        return pd.concat(frames, ignore_index=True), reports
    return pd.DataFrame(columns=list(LM_REQUIRED_OUTPUT_COLUMNS)), reports


def score_word_table(frame: pd.DataFrame, config: LMScoringConfig) -> LMScoringResult:
    """Score a word table and return stimulus-level LM features plus output frame."""

    if config.shard_id < 0 or config.num_shards <= 0 or config.shard_id >= config.num_shards:
        raise ValueError("shard_id must be in [0, num_shards)")
    contexts = group_word_rows_for_lm(frame, config.context_mode)
    manifest: dict[str, Any] = {
        "run_type": "score_predictability",
        "model_id": MOCK_MODEL_ID if config.mock_model else config.model_id,
        "reference_model_id": config.model_id,
        "tokenizer_id": MOCK_MODEL_ID if config.mock_model else (config.tokenizer_id or config.model_id),
        "context_mode": config.context_mode,
        "device": config.device,
        "dtype": config.dtype,
        "mock_model": bool(config.mock_model),
        "dry_run": bool(config.dry_run),
        "log_base": config.log_base,
        "surprisal_units": "natural_log",
        "entropy_units": "natural_log",
        "input_rows": int(len(frame)),
        "contexts_total": int(len(contexts)),
        "shard_id": int(config.shard_id),
        "num_shards": int(config.num_shards),
    }
    if config.dry_run:
        stimulus = prepare_stimulus_word_table(frame, config.context_mode)
        manifest["status"] = "dry_run_complete"
        manifest["stimulus_rows"] = int(len(stimulus))
        return LMScoringResult(stimulus, pd.DataFrame(), manifest, {"status": "not_run", "reports": []})
    lm_features, reports = _score_contexts(contexts, config)
    ordered_columns = [
        "lm_stable_word_id",
        "speech_id",
        "paragraph_id",
        "sentence_id",
        "source_word_id",
        "word",
        "lm_model_id",
        "lm_tokenizer_id",
        "lm_context_mode",
        "lm_context_id",
        "lm_context_tokens",
        "lm_word_surprisal",
        "lm_word_entropy",
        "lm_word_entropy_onset",
        "lm_subword_count",
        "lm_scored_subword_count",
        "lm_alignment_status",
        "lm_alignment_warning",
        "lm_alignment_error",
    ]
    lm_features = lm_features[[column for column in ordered_columns if column in lm_features.columns]]
    validate_lm_feature_table(lm_features).raise_for_errors()
    output = _merge_if_repeated_input(frame, lm_features)
    alignment_report = {
        "status": "passed" if all(report["status"] in {"ok", "warning"} for report in reports) else "failed",
        "reports": reports,
    }
    manifest["status"] = "complete"
    manifest["lm_feature_rows"] = int(len(lm_features))
    manifest["output_rows"] = int(len(output))
    manifest["output_columns"] = list(output.columns)
    return LMScoringResult(output, lm_features, manifest, alignment_report)


def score_word_table_mock(frame: pd.DataFrame, config: LMScoringConfig | None = None) -> LMScoringResult:
    """Score a word table with deterministic local token scores."""

    cfg = config or LMScoringConfig(mock_model=True)
    cfg = LMScoringConfig(
        model_id=cfg.model_id,
        tokenizer_id=cfg.tokenizer_id,
        context_mode=cfg.context_mode,
        device=cfg.device,
        dtype=cfg.dtype,
        mock_model=True,
        dry_run=cfg.dry_run,
        shard_id=cfg.shard_id,
        num_shards=cfg.num_shards,
        log_base=cfg.log_base,
    )
    return score_word_table(frame, cfg)


def validate_lm_feature_table(frame: pd.DataFrame) -> LMOutputValidationResult:
    """Validate word-level LM feature output schema."""

    errors: list[str] = []
    missing = [column for column in LM_REQUIRED_OUTPUT_COLUMNS if column not in frame.columns]
    if missing:
        errors.append("missing LM output columns: " + ", ".join(missing))
    if not missing:
        if frame[LM_STABLE_WORD_ID].isna().any():
            errors.append("lm_stable_word_id contains null values")
        if frame[LM_STABLE_WORD_ID].duplicated().any():
            errors.append("lm_stable_word_id must be unique in stimulus-level LM output")
        for column in ("lm_word_surprisal", "lm_word_entropy", "lm_word_entropy_onset"):
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any():
                errors.append(f"{column} contains missing or non-numeric values")
            if (values < 0).any():
                errors.append(f"{column} contains negative values")
        counts = pd.to_numeric(frame["lm_subword_count"], errors="coerce")
        scored = pd.to_numeric(frame["lm_scored_subword_count"], errors="coerce")
        if counts.isna().any() or (counts <= 0).any():
            errors.append("lm_subword_count must be positive")
        if scored.isna().any() or (scored <= 0).any():
            errors.append("lm_scored_subword_count must be positive")
        bad_status = set(frame["lm_alignment_status"].dropna().astype(str)) - {"ok", "warning"}
        if bad_status:
            errors.append("lm_alignment_status contains invalid values: " + ", ".join(sorted(bad_status)))
    return LMOutputValidationResult(ok=not errors, errors=tuple(errors))


def write_lm_feature_outputs(
    result: LMScoringResult,
    out_path: str | Path,
    manifest_path: str | Path | None = None,
    alignment_report_path: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write scored table and sidecar JSON reports."""

    output = Path(out_path)
    manifest = Path(manifest_path) if manifest_path else output.with_suffix(".manifest.json")
    alignment = Path(alignment_report_path) if alignment_report_path else output.with_suffix(".alignment_report.json")
    write_table(result.frame, output)
    payload = dict(result.manifest)
    payload["output_path"] = str(output)
    payload["alignment_report_path"] = str(alignment)
    _write_json(manifest, payload)
    _write_json(alignment, result.alignment_report)
    return output, manifest, alignment


def write_scoring_result(result: LMScoringResult, output_path: str | Path) -> tuple[Path, Path, Path]:
    """Compatibility wrapper for writing score outputs."""

    return write_lm_feature_outputs(result, output_path)
