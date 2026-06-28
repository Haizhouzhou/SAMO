import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_help_runs():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run([sys.executable, "-m", "samo_copco.cli", "--help"], cwd=root, text=True, capture_output=True, env=env)
    assert result.returncode == 0
    assert "run-synthetic" in result.stdout


def test_synthetic_cli_smoke_run_writes_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    out = tmp_path / "synthetic_run"
    result = subprocess.run([sys.executable, "-m", "samo_copco.cli", "run-synthetic", "--out", str(out)], cwd=root, text=True, capture_output=True, env=env)
    assert result.returncode == 0, result.stderr
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_readers"] == 8
    assert (out / "reader_profiles.csv").stat().st_size > 0
