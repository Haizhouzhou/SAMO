import csv
import subprocess
import sys
from pathlib import Path


def _release_files(root: Path):
    ignored = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "build", "dist"}
    return sorted(path for path in root.rglob("*") if path.is_file() and not any(part in ignored for part in path.relative_to(root).parts))


def test_public_release_sanitizer_passes():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "scripts/check_public_release.py"], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_manifest_matches_public_file_count():
    root = Path(__file__).resolve().parents[1]
    rows = list(csv.DictReader((root / "PUBLIC_RELEASE_MANIFEST.tsv").read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    assert len(rows) == len(_release_files(root))


def test_no_outside_symlinks_and_no_data_like_extensions():
    root = Path(__file__).resolve().parents[1]
    forbidden_suffixes = {".edf", ".asc", ".ias", ".dat", ".xlsx", ".parquet", ".pkl", ".pickle", ".pt", ".pth", ".npy", ".npz", ".onnx"}
    for path in root.rglob("*"):
        if ".git" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            path.resolve().relative_to(root.resolve())
        if path.is_file():
            assert path.suffix.lower() not in forbidden_suffixes
