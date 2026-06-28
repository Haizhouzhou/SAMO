from pathlib import Path

import pandas as pd

from samo_copco.data import prepare_word_table
from samo_copco.profiles import build_reader_profiles
from samo_copco.residualization import FoldLocalResidualizer


def test_reader_profiles_one_row_per_reader():
    root = Path(__file__).resolve().parents[1]
    word = pd.read_csv(root / "tests" / "fixtures" / "synthetic_word_features.csv")
    labels = pd.read_csv(root / "tests" / "fixtures" / "synthetic_labels.csv")
    prepared = prepare_word_table(word, labels)
    transformed = FoldLocalResidualizer().fit(prepared).transform(prepared)
    profiles = build_reader_profiles(transformed)
    assert len(profiles) == prepared["reader_id"].nunique()
    assert not profiles["reader_id"].duplicated().any()
    assert "predictability_residual_slope" in profiles.columns
