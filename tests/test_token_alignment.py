from __future__ import annotations

import pytest

from samo_copco.token_alignment import (
    TokenSpan,
    WordSpan,
    aggregate_token_scores_to_words,
    align_token_offsets_to_word_spans,
    validate_token_word_alignment,
)


def test_token_offsets_align_by_maximum_overlap_and_ignore_special_tokens() -> None:
    words = [WordSpan("w1", 0, 3, "Hej"), WordSpan("w2", 4, 10, "verden")]
    assignments = align_token_offsets_to_word_spans(
        [TokenSpan(0, 0, 0, True), TokenSpan(1, 0, 2), TokenSpan(2, 2, 5), TokenSpan(3, 7, 10)],
        words,
    )
    assert assignments[0].word_span is None
    assert assignments[1].lm_stable_word_id == "w1"
    assert assignments[2].lm_stable_word_id == "w1"
    assert assignments[3].lm_stable_word_id == "w2"


def test_alignment_reports_unassigned_non_special_tokens() -> None:
    words = [WordSpan("w1", 0, 3, "Hej")]
    assignments = align_token_offsets_to_word_spans([TokenSpan(0, 0, 3), TokenSpan(1, 8, 10)], words)
    report = validate_token_word_alignment(assignments, words)
    assert report["status"] == "warning"
    assert "non_special_token_unassigned" in report["warnings"]


def test_word_aggregation_sums_surprisal_and_averages_entropy() -> None:
    words = [WordSpan("w1", 0, 3, "Hej"), WordSpan("w2", 4, 10, "verden")]
    assignments = align_token_offsets_to_word_spans(
        [TokenSpan(0, 0, 0, True), TokenSpan(1, 0, 3), TokenSpan(2, 4, 7), TokenSpan(3, 7, 10)],
        words,
    )
    rows = aggregate_token_scores_to_words(
        assignments,
        {
            1: {"surprisal": 0.5, "entropy": 2.0},
            2: {"surprisal": 1.0, "entropy": 3.0},
            3: {"surprisal": 1.5, "entropy": 5.0},
        },
        words,
    )
    by_word = rows.set_index("lm_stable_word_id")
    assert by_word.loc["w1", "lm_word_surprisal"] == pytest.approx(0.5)
    assert by_word.loc["w1", "lm_word_entropy"] == pytest.approx(2.0)
    assert by_word.loc["w2", "lm_word_surprisal"] == pytest.approx(2.5)
    assert by_word.loc["w2", "lm_word_entropy"] == pytest.approx(4.0)
    assert by_word.loc["w2", "lm_word_entropy_onset"] == pytest.approx(3.0)
