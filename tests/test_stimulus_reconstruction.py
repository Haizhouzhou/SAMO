from __future__ import annotations

import pandas as pd

from samo_copco.stimulus_reconstruction import group_word_rows_for_lm, make_lm_stable_word_id, prepare_stimulus_word_table


def test_stimulus_reconstruction_builds_stable_ids_and_context_text() -> None:
    frame = pd.DataFrame(
        {
            "speechId": [100, 100],
            "paragraphId": [0, 0],
            "sentenceId": [0, 0],
            "wordId": [0, 1],
            "word": ["Hej", "verden"],
        }
    )
    normalized = make_lm_stable_word_id(frame)
    assert normalized["lm_stable_word_id"].tolist() == ["100::0::0::0", "100::0::0::1"]
    contexts = group_word_rows_for_lm(frame, "paragraph")
    assert len(contexts) == 1
    assert contexts[0].context_text == "Hej verden"
    assert contexts[0].word_spans[1].start == 4


def test_prepare_stimulus_does_not_collapse_local_wordid_across_paragraphs() -> None:
    frame = pd.DataFrame(
        {
            "speechId": [100, 100],
            "paragraphId": [0, 1],
            "sentenceId": [0, 0],
            "wordId": [0, 0],
            "word": ["Alpha", "Gamma"],
        }
    )
    stimulus = prepare_stimulus_word_table(frame)
    assert len(stimulus) == 2
    assert stimulus["lm_stable_word_id"].nunique() == 2
