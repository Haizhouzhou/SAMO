"""Tokenizer/subword alignment to CopCo word spans."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class WordSpan:
    """A word span in reconstructed context text."""

    lm_stable_word_id: str
    start: int
    end: int
    word: str | None = None
    speech_id: str | None = None
    paragraph_id: str | None = None
    sentence_id: str | None = None
    source_word_id: str | None = None
    row_index: int | None = None

    @property
    def word_id(self) -> str:
        return self.lm_stable_word_id

    @property
    def word_form(self) -> str | None:
        return self.word


@dataclass(frozen=True)
class TokenSpan:
    """A tokenizer offset span."""

    token_index: int
    start: int
    end: int
    is_special: bool = False


@dataclass(frozen=True)
class TokenWordAssignment:
    """A token-to-word assignment by character overlap."""

    token_span: TokenSpan
    word_span: WordSpan | None
    overlap: int

    @property
    def lm_stable_word_id(self) -> str | None:
        return None if self.word_span is None else self.word_span.lm_stable_word_id


def _as_token_spans(token_offsets: Iterable[tuple[int, int] | TokenSpan]) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    for index, item in enumerate(token_offsets):
        if isinstance(item, TokenSpan):
            spans.append(item)
        else:
            start, end = item
            spans.append(TokenSpan(index, int(start), int(end), int(end) <= int(start)))
    return spans


def align_token_offsets_to_word_spans(
    token_offsets: Iterable[tuple[int, int] | TokenSpan],
    word_spans: Iterable[WordSpan],
) -> list[TokenWordAssignment]:
    """Assign each non-special token offset to the word with maximum overlap."""

    words = list(word_spans)
    assignments: list[TokenWordAssignment] = []
    for token in _as_token_spans(token_offsets):
        if token.is_special or token.end <= token.start:
            assignments.append(TokenWordAssignment(token, None, 0))
            continue
        best_word: WordSpan | None = None
        best_overlap = 0
        for word in words:
            overlap = max(0, min(token.end, word.end) - max(token.start, word.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_word = word
        assignments.append(TokenWordAssignment(token, best_word if best_overlap > 0 else None, best_overlap))
    return assignments


def validate_token_word_alignment(
    assignments: Iterable[TokenWordAssignment],
    word_spans: Iterable[WordSpan],
) -> dict[str, Any]:
    """Validate token-word assignments and report alignment status."""

    words = list(word_spans)
    assigned = list(assignments)
    errors: list[str] = []
    warnings: list[str] = []
    ids = [span.lm_stable_word_id for span in words]
    missing_ids = [word_id for word_id in ids if not str(word_id)]
    duplicates = [word_id for word_id, count in Counter(ids).items() if count > 1]
    if missing_ids:
        errors.append(f"missing_stable_word_ids:{len(missing_ids)}")
    if duplicates:
        errors.append(f"duplicate_stable_word_ids:{len(duplicates)}")
    counts = {word_id: 0 for word_id in ids}
    for assignment in assigned:
        token = assignment.token_span
        if token.is_special or token.end <= token.start:
            continue
        if assignment.word_span is None:
            warnings.append("non_special_token_unassigned")
            continue
        counts[assignment.word_span.lm_stable_word_id] += 1
    zero_words = [word_id for word_id, count in counts.items() if count == 0]
    if zero_words:
        errors.append(f"zero_subword_words:{len(zero_words)}")
    deduped_warnings = sorted(set(warnings))
    return {
        "status": "error" if errors else ("warning" if deduped_warnings else "ok"),
        "errors": errors,
        "warnings": deduped_warnings,
        "word_count": len(words),
        "token_count": len(assigned),
        "word_subword_counts": counts,
    }


def aggregate_token_scores_to_words(
    assignments: Iterable[TokenWordAssignment],
    token_scores: dict[int, dict[str, float]],
    word_spans: Iterable[WordSpan],
) -> pd.DataFrame:
    """Aggregate shifted token scores to word-level surprisal and entropy."""

    words = list(word_spans)
    report = validate_token_word_alignment(assignments, words)
    values: dict[str, dict[str, Any]] = {
        word.lm_stable_word_id: {
            "lm_stable_word_id": word.lm_stable_word_id,
            "speech_id": word.speech_id,
            "paragraph_id": word.paragraph_id,
            "sentence_id": word.sentence_id,
            "source_word_id": word.source_word_id,
            "word": word.word,
            "lm_word_surprisal": 0.0,
            "lm_word_entropy_sum": 0.0,
            "lm_word_entropy_onset": float("nan"),
            "lm_subword_count": int(report["word_subword_counts"].get(word.lm_stable_word_id, 0)),
            "lm_scored_subword_count": 0,
        }
        for word in words
    }
    for assignment in assignments:
        word = assignment.word_span
        if word is None:
            continue
        score = token_scores.get(assignment.token_span.token_index)
        if score is None:
            continue
        row = values[word.lm_stable_word_id]
        surprisal = float(score["surprisal"])
        entropy = float(score["entropy"])
        row["lm_word_surprisal"] += surprisal
        row["lm_word_entropy_sum"] += entropy
        row["lm_scored_subword_count"] += 1
        if pd.isna(row["lm_word_entropy_onset"]):
            row["lm_word_entropy_onset"] = entropy
    status = report["status"]
    warning = ";".join(report["warnings"]) or None
    error = ";".join(report["errors"]) or None
    rows: list[dict[str, Any]] = []
    for row in values.values():
        scored = int(row["lm_scored_subword_count"])
        entropy = row["lm_word_entropy_sum"] / scored if scored else float("nan")
        rows.append(
            {
                "lm_stable_word_id": row["lm_stable_word_id"],
                "speech_id": row["speech_id"],
                "paragraph_id": row["paragraph_id"],
                "sentence_id": row["sentence_id"],
                "source_word_id": row["source_word_id"],
                "word": row["word"],
                "lm_word_surprisal": row["lm_word_surprisal"] if scored else float("nan"),
                "lm_word_entropy": entropy,
                "lm_word_entropy_onset": row["lm_word_entropy_onset"],
                "lm_subword_count": int(row["lm_subword_count"]),
                "lm_scored_subword_count": scored,
                "lm_alignment_status": status,
                "lm_alignment_warning": warning,
                "lm_alignment_error": error,
            }
        )
    return pd.DataFrame(rows)
