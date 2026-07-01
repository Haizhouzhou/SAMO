"""CopCo stimulus normalization and context reconstruction for LM scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data_contracts import (
    LM_STABLE_WORD_ID,
    PARAGRAPH_ID,
    READER_ID,
    SENTENCE_ID,
    SOURCE_WORD_ID,
    SPEECH_ID,
    TEXT_ID,
    TRIAL_ID,
    WORD,
    WORD_ID,
    WORD_POSITION,
)
from .features import normalize_feature_columns
from .token_alignment import WordSpan


@dataclass(frozen=True)
class StimulusContext:
    """Reconstructed text context and its word spans."""

    context_id: str
    context_mode: str
    context_text: str
    word_spans: tuple[WordSpan, ...]

    @property
    def text(self) -> str:
        return self.context_text

    @property
    def spans(self) -> tuple[WordSpan, ...]:
        return self.word_spans


def normalize_copco_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize public and CopCo camelCase columns to SAMO snake_case names."""

    out = normalize_feature_columns(frame)
    if WORD not in out.columns:
        raise ValueError("word table requires a word column")
    if SOURCE_WORD_ID not in out.columns and WORD_ID in out.columns:
        out[SOURCE_WORD_ID] = out[WORD_ID].astype(str)
    if WORD_POSITION not in out.columns:
        group_cols = [column for column in (SPEECH_ID, PARAGRAPH_ID, SENTENCE_ID, TEXT_ID) if column in out.columns]
        if group_cols:
            out[WORD_POSITION] = out.groupby(group_cols, sort=False).cumcount() + 1
        else:
            out[WORD_POSITION] = range(1, len(out) + 1)
    return out


def _string_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("string").fillna("").astype(str)


def make_lm_stable_word_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Add stable stimulus word IDs without treating local source_word_id as global."""

    out = normalize_copco_columns(frame)
    if {SPEECH_ID, PARAGRAPH_ID, SENTENCE_ID, SOURCE_WORD_ID} <= set(out.columns):
        out[LM_STABLE_WORD_ID] = (
            _string_column(out, SPEECH_ID)
            + "::"
            + _string_column(out, PARAGRAPH_ID)
            + "::"
            + _string_column(out, SENTENCE_ID)
            + "::"
            + _string_column(out, SOURCE_WORD_ID)
        )
    elif {SPEECH_ID, PARAGRAPH_ID, SOURCE_WORD_ID} <= set(out.columns):
        out[LM_STABLE_WORD_ID] = (
            _string_column(out, SPEECH_ID)
            + "::"
            + _string_column(out, PARAGRAPH_ID)
            + "::"
            + _string_column(out, SOURCE_WORD_ID)
        )
    elif WORD_ID in out.columns:
        out[LM_STABLE_WORD_ID] = _string_column(out, WORD_ID)
        if SOURCE_WORD_ID not in out.columns:
            out[SOURCE_WORD_ID] = _string_column(out, WORD_ID)
    elif LM_STABLE_WORD_ID not in out.columns:
        raise ValueError("cannot construct lm_stable_word_id without CopCo IDs or globally unique word_id")
    if WORD_ID not in out.columns:
        out[WORD_ID] = out[LM_STABLE_WORD_ID]
    return out


def _sort_columns(frame: pd.DataFrame, *, include_reader: bool = False) -> list[str]:
    candidates = [
        READER_ID if include_reader else "",
        SPEECH_ID,
        PARAGRAPH_ID,
        SENTENCE_ID,
        TEXT_ID,
        TRIAL_ID,
        WORD_POSITION,
        SOURCE_WORD_ID,
        WORD_ID,
        "_source_row_index",
    ]
    return [column for column in candidates if column and column in frame.columns]


def _context_column(frame: pd.DataFrame, context_mode: str) -> str:
    if context_mode == "paragraph":
        if {SPEECH_ID, PARAGRAPH_ID} <= set(frame.columns):
            return "_lm_paragraph_context_id"
        if PARAGRAPH_ID in frame.columns:
            return PARAGRAPH_ID
        if TEXT_ID in frame.columns:
            return TEXT_ID
    if context_mode == "sentence":
        if {SPEECH_ID, PARAGRAPH_ID, SENTENCE_ID} <= set(frame.columns):
            return "_lm_sentence_context_id"
        if SENTENCE_ID in frame.columns:
            return SENTENCE_ID
        if PARAGRAPH_ID in frame.columns:
            return PARAGRAPH_ID
    if context_mode == "text":
        if TEXT_ID in frame.columns:
            return TEXT_ID
        if SPEECH_ID in frame.columns:
            return SPEECH_ID
    raise ValueError("context_mode must be paragraph, sentence, or text and needs matching columns")


def prepare_stimulus_word_table(frame: pd.DataFrame, context_mode: str = "paragraph") -> pd.DataFrame:
    """Return one stimulus row per stable word ID with context IDs and ordering."""

    out = make_lm_stable_word_id(frame).copy().reset_index(drop=False).rename(columns={"index": "_source_row_index"})
    if {SPEECH_ID, PARAGRAPH_ID} <= set(out.columns):
        out["_lm_paragraph_context_id"] = _string_column(out, SPEECH_ID) + "::" + _string_column(out, PARAGRAPH_ID)
    if {SPEECH_ID, PARAGRAPH_ID, SENTENCE_ID} <= set(out.columns):
        out["_lm_sentence_context_id"] = (
            _string_column(out, SPEECH_ID)
            + "::"
            + _string_column(out, PARAGRAPH_ID)
            + "::"
            + _string_column(out, SENTENCE_ID)
        )
    context_col = _context_column(out, context_mode)
    sort_cols = _sort_columns(out)
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    conflicts = out.groupby(LM_STABLE_WORD_ID, dropna=False)[WORD].nunique(dropna=False)
    bad = conflicts[conflicts > 1]
    if not bad.empty:
        raise ValueError("lm_stable_word_id maps to multiple word forms")
    keep_cols = [column for column in out.columns if column != READER_ID]
    stimulus = out[keep_cols].drop_duplicates(LM_STABLE_WORD_ID, keep="first").copy()
    stimulus["_lm_context_id"] = stimulus[context_col].astype(str)
    return stimulus.sort_values(_sort_columns(stimulus), kind="mergesort").reset_index(drop=True)


def rebuild_text_from_word_rows(rows: pd.DataFrame, word_column: str = WORD) -> str:
    """Rebuild one-space-normalized context text from ordered word rows."""

    return " ".join(rows[word_column].astype(str).tolist())


def make_word_spans(words: pd.DataFrame) -> list[WordSpan]:
    """Create word spans for a reconstructed context."""

    spans: list[WordSpan] = []
    position = 0
    for ordinal, row in enumerate(words.itertuples(index=False)):
        word = str(getattr(row, WORD))
        start = position
        end = start + len(word)
        stable_id = str(getattr(row, LM_STABLE_WORD_ID))
        spans.append(
            WordSpan(
                lm_stable_word_id=stable_id,
                start=start,
                end=end,
                word=word,
                speech_id=str(getattr(row, SPEECH_ID)) if hasattr(row, SPEECH_ID) else None,
                paragraph_id=str(getattr(row, PARAGRAPH_ID)) if hasattr(row, PARAGRAPH_ID) else None,
                sentence_id=str(getattr(row, SENTENCE_ID)) if hasattr(row, SENTENCE_ID) else None,
                source_word_id=str(getattr(row, SOURCE_WORD_ID)) if hasattr(row, SOURCE_WORD_ID) else None,
                row_index=ordinal,
            )
        )
        position = end + 1
    return spans


def group_word_rows_for_lm(frame: pd.DataFrame, context_mode: str = "paragraph") -> list[StimulusContext]:
    """Group normalized stimulus rows into LM contexts."""

    stimulus = prepare_stimulus_word_table(frame, context_mode=context_mode)
    contexts: list[StimulusContext] = []
    for context_id, group in stimulus.groupby("_lm_context_id", sort=False):
        ordered = group.sort_values(_sort_columns(group), kind="mergesort").reset_index(drop=True)
        contexts.append(
            StimulusContext(
                context_id=str(context_id),
                context_mode=context_mode,
                context_text=rebuild_text_from_word_rows(ordered),
                word_spans=tuple(make_word_spans(ordered)),
            )
        )
    return contexts


def reconstruct_stimulus_contexts(frame: pd.DataFrame, *, context_mode: str = "paragraph") -> list[StimulusContext]:
    """Compatibility wrapper for context reconstruction."""

    return group_word_rows_for_lm(frame, context_mode=context_mode)
