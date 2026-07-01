from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_score_predictability_mock_cli_writes_required_lm_columns(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    output = tmp_path / "mock_lm_features.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "samo_copco.cli",
            "score-predictability",
            "--input",
            "tests/fixtures/synthetic_lm_words.csv",
            "--out",
            str(output),
            "--mock-model",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(output)
    required = {
        "lm_stable_word_id",
        "word",
        "lm_word_surprisal",
        "lm_word_entropy",
        "lm_word_entropy_onset",
        "lm_subword_count",
        "lm_scored_subword_count",
        "lm_alignment_status",
    }
    assert required <= set(frame.columns)
    assert set(frame["lm_alignment_status"]) == {"ok"}
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["mock_model"] is True
    assert manifest["surprisal_units"] == "natural_log"
