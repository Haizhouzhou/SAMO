from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_public_release_sanitizer_passes() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "scripts/check_public_release.py"], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_no_data_docs_notebooks_or_public_audit_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden_suffixes = {".edf", ".asc", ".ias", ".dat", ".xlsx", ".parquet", ".pkl", ".pickle", ".pt", ".pth", ".npy", ".npz", ".onnx", ".pdf", ".ipynb"}
    assert not (root / "docs").exists()
    assert not (root / "PUBLIC_RELEASE_MANIFEST.tsv").exists()
    assert not (root / "RELEASE_BUILD_REPORT.md").exists()
    for path in root.rglob("*"):
        if any(part in {".git", "__pycache__", ".pytest_cache", "build", "dist"} for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            assert path.suffix.lower() not in forbidden_suffixes
