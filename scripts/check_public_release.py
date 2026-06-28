from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path


def _text_from_codes(codes: list[int]) -> str:
    return "".join(chr(code) for code in codes)


FORBIDDEN_TEXT = [
    _text_from_codes([68, 51]),
    _text_from_codes([68, 51, 95, 76, 105, 116, 101]),
    _text_from_codes([66, 101, 110, 99, 104, 109, 97, 114, 107, 66, 114, 105, 100, 103, 101]),
    _text_from_codes([79, 102, 102, 105, 99, 105, 97, 108, 69, 121, 101, 66, 101, 110, 99, 104, 65, 108, 105, 103, 110, 109, 101, 110, 116]),
    _text_from_codes([79, 112, 101, 114, 97, 116, 105, 110, 103, 80, 111, 105, 110, 116, 65, 100, 97, 112, 116, 97, 116, 105, 111, 110]),
    _text_from_codes([79, 110, 108, 105, 110, 101, 84, 97, 114, 103, 101, 116, 101, 100, 79, 112, 116, 105, 109, 105, 122, 97, 116, 105, 111, 110]),
    _text_from_codes([68, 51, 77, 111, 100, 101, 108, 69, 118, 105, 100, 101, 110, 99, 101, 86, 97, 117, 108, 116]),
    _text_from_codes([114, 101, 115, 99, 117, 101, 95, 48, 52]),
    _text_from_codes([114, 101, 115, 99, 117, 101, 95, 48, 53]),
    _text_from_codes([67, 111, 100, 101, 120]),
    _text_from_codes([71, 105, 122, 109, 111]),
    _text_from_codes([104, 97, 105, 122, 104, 101]),
    _text_from_codes([108, 105, 110, 117, 120]),
    _text_from_codes([117, 50, 52, 45, 108, 111, 103, 105, 110]),
    _text_from_codes([68, 69, 83, 75, 84, 79, 80]),
    _text_from_codes([72, 69, 73, 90, 79, 85]),
    _text_from_codes([47, 104, 111, 109, 101, 47]),
    _text_from_codes([47, 109, 110, 116, 47]),
]

DATA_EXTENSIONS = {
    ".edf",
    ".asc",
    ".ias",
    ".dat",
    ".xlsx",
    ".parquet",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
    ".onnx",
}
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "build", "dist"}
LOCAL_DATA_PARTS = {"data", "raw", "external", "extracted", "derived", "results", "outputs", "logs"}
ALLOWED_SYNTHETIC_TABLES = {
    Path("tests/fixtures/synthetic_word_features.csv"),
    Path("tests/fixtures/synthetic_labels.csv"),
}
SECRET_TEXT = ["tok" + "en", "pass" + "word", "api" + "_" + "key", "private" + "_" + "key"]
STUB_TEXT = ["TO" + "DO", "FIX" + "ME", "Not" + "Implemented" + "Error"]


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in rel.parts):
            continue
        if path.is_file() or path.is_symlink():
            yield path


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _is_inside(child: Path, root: Path) -> bool:
    try:
        child.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def check_manifest(root: Path, errors: list[str]) -> None:
    manifest = root / "PUBLIC_RELEASE_MANIFEST.tsv"
    if not manifest.exists():
        errors.append("missing public manifest")
        return
    rows = list(csv.DictReader(manifest.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    rels = {str(path.relative_to(root)) for path in iter_files(root)}
    manifest_rels = {row.get("relative_path", "") for row in rows}
    if len(rows) != len(rels):
        errors.append(f"manifest row count {len(rows)} does not equal file count {len(rels)}")
    missing = sorted(rels - manifest_rels)
    extra = sorted(manifest_rels - rels)
    if missing:
        errors.append("files missing from manifest: " + ", ".join(missing[:10]))
    if extra:
        errors.append("manifest rows without files: " + ", ".join(extra[:10]))
    required = {"relative_path", "size_bytes", "sha256", "file_role", "why_public", "source_logic_basis", "data_safety_status", "claim_safety_status"}
    if rows and set(rows[0]) != required:
        errors.append("manifest columns do not match required schema")


def check_notebooks(path: Path, text: str, errors: list[str]) -> None:
    if path.suffix != ".ipynb":
        return
    payload = json.loads(text)
    for cell in payload.get("cells", []):
        if cell.get("outputs"):
            errors.append(f"notebook has outputs: {path}")
            return


def check_tables(root: Path, path: Path, text: str, errors: list[str]) -> None:
    rel = path.relative_to(root)
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return
    if rel in ALLOWED_SYNTHETIC_TABLES:
        if "synthetic_" not in text:
            errors.append(f"synthetic fixture lacks synthetic marker: {rel}")
        return
    header = text.splitlines()[0].lower() if text.splitlines() else ""
    if "reader_id" in header and "reader_label" in header:
        errors.append(f"reader-labelled table outside synthetic fixtures: {rel}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors: list[str] = []
    for path in iter_files(root):
        rel = path.relative_to(root)
        if path.is_symlink():
            target = path.resolve()
            if not _is_inside(target, root):
                errors.append(f"symlink points outside repo: {rel}")
            continue
        if path.suffix.lower() in DATA_EXTENSIONS:
            errors.append(f"forbidden data-like extension: {rel}")
        if path.stat().st_size > 1_000_000:
            errors.append(f"file larger than one megabyte: {rel}")
        if any(part in LOCAL_DATA_PARTS for part in rel.parts):
            errors.append(f"local-data folder present: {rel}")
        text = _read_text(path)
        if text is None:
            continue
        for pattern in FORBIDDEN_TEXT:
            if pattern in str(rel) or pattern in text:
                errors.append(f"forbidden string found in {rel}")
                break
        lowered = text.lower()
        for pattern in SECRET_TEXT:
            if pattern in lowered:
                errors.append(f"secret-like pattern found in {rel}")
                break
        for pattern in STUB_TEXT:
            if pattern in text:
                errors.append(f"stub marker found in {rel}")
                break
        if rel.parts[:2] == ("src", "samo_copco") and path.suffix == ".py":
            if re.search(r"(?m)^\s*pass\s*(#.*)?$", text):
                errors.append(f"pass-only source line found in {rel}")
        check_notebooks(path, text, errors)
        check_tables(root, path, text, errors)
    check_manifest(root, errors)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("public release sanitizer passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
