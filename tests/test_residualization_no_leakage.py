from pathlib import Path

import pandas as pd
import pytest

from samo_copco.data import prepare_word_table
from samo_copco.residualization import FoldLocalResidualizer


def _prepared():
    root = Path(__file__).resolve().parents[1]
    word = pd.read_csv(root / "tests" / "fixtures" / "synthetic_word_features.csv")
    labels = pd.read_csv(root / "tests" / "fixtures" / "synthetic_labels.csv")
    return prepare_word_table(word, labels)


def test_residualizer_rejects_heldout_reader_rows_during_fit():
    frame = _prepared()
    heldout = {"synthetic_reader_01"}
    with pytest.raises(ValueError):
        FoldLocalResidualizer().fit(frame, heldout_readers=heldout)


def test_residualizer_fit_readers_are_training_only():
    frame = _prepared()
    heldout = {"synthetic_reader_01"}
    train = frame[~frame["reader_id"].isin(heldout)]
    residualizer = FoldLocalResidualizer().fit(train, heldout_readers=heldout)
    assert heldout.isdisjoint(residualizer.fit_reader_ids_)
    transformed = residualizer.transform(frame)
    assert "gaze_residual" in transformed.columns
