from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_help_and_validate_dry_run() -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    help_result = subprocess.run([sys.executable, "-m", "samo_copco.cli", "--help"], cwd=root, text=True, capture_output=True, env=env)
    assert help_result.returncode == 0
    assert "score-predictability" in help_result.stdout
    validate = subprocess.run(
        [sys.executable, "-m", "samo_copco.cli", "validate-data", "--config", "configs/copco_paths.example.yaml", "--dry-run"],
        cwd=root,
        text=True,
        capture_output=True,
        env=env,
    )
    assert validate.returncode == 0
    assert "dry-run ok" in validate.stdout


def test_synthetic_pipeline_writes_metrics_and_profiles(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    out = tmp_path / "synthetic"
    result = subprocess.run([sys.executable, "-m", "samo_copco.cli", "run-synthetic", "--out", str(out)], cwd=root, text=True, capture_output=True, env=env)
    assert result.returncode == 0, result.stderr
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_readers"] == 8
    assert (out / "reader_profiles.csv").stat().st_size > 0
