from __future__ import annotations

import math

import numpy as np
import pytest

from samo_copco.lm_scoring import causal_next_token_scores


def test_shifted_next_token_surprisal_and_entropy_use_natural_logs() -> None:
    logits = np.array([[[0.0, 1.0, 2.0], [3.0, 0.0, -1.0], [0.5, 0.5, 0.5]]])
    input_ids = np.array([[0, 2, 1]])
    surprisal, entropy = causal_next_token_scores(logits, input_ids)

    first_log_probs = logits[0, 0] - math.log(float(np.exp(logits[0, 0]).sum()))
    second_log_probs = logits[0, 1] - math.log(float(np.exp(logits[0, 1]).sum()))
    assert surprisal == pytest.approx([-first_log_probs[2], -second_log_probs[1]])

    first_probs = np.exp(first_log_probs)
    second_probs = np.exp(second_log_probs)
    assert entropy == pytest.approx(
        [
            -float((first_probs * first_log_probs).sum()),
            -float((second_probs * second_log_probs).sum()),
        ]
    )
    assert surprisal[0] == pytest.approx(-math.log(math.exp(2.0) / np.exp(logits[0, 0]).sum()))
