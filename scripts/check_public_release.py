from __future__ import annotations

import re
import sys
from pathlib import Path


def _text(codes: list[int]) -> str:
    return "".join(chr(code) for code in codes)


FORBIDDEN_TEXT = [
    _text([104, 97, 105, 122, 104, 101]),
    _text([117, 50, 52, 45, 108, 111, 103, 105, 110]),
    _text([68, 69, 83, 75, 84, 79, 80]),
    _text([72, 69, 73, 90, 79, 85]),
    _text([47, 104, 111, 109, 101, 47]),
    _text([47, 109, 110, 116, 47]),
    _text([68, 51, 95, 76, 105, 116, 101]),
    _text([66, 101, 110, 99, 104, 109, 97, 114, 107, 66, 114, 105, 100, 103, 101]),
    _text([79, 102, 102, 105, 99, 105, 97, 108, 69, 121, 101, 66, 101, 110, 99, 104, 65, 108, 105, 103, 110, 109, 101, 110, 116]),
    _text([79, 112, 101, 114, 97, 116, 105, 110, 103, 80, 111, 105, 110, 116, 65, 100, 97, 112, 116, 97, 116, 105, 111, 110]),
    _text([79, 110, 108, 105, 110, 101, 84, 97, 114, 103, 101, 116, 101, 100, 79, 112, 116, 105, 109, 105, 122, 97, 116, 105, 111, 110]),
    _text([68, 51, 77, 111, 100, 101, 108, 69, 118, 105, 100, 101, 110, 99, 101, 86, 97, 117, 108, 116]),
    _text([114, 101, 115, 99, 117, 101, 95, 48, 52]),
    _text([114, 101, 115, 99, 117, 101, 95, 48, 53]),
    _text([67, 111, 100, 101, 120]),
    _text([71, 105, 122, 109, 111]),
    _text([60, 80, 85, 84, 95, 76, 73, 67, 69, 78, 83, 69, 95, 72, 69, 82, 69, 62]),
    _text([80, 85, 84, 95, 76, 73, 67, 69, 78, 83, 69, 95, 72, 69, 82, 69]),
    _text([76, 73, 67, 69, 78, 83, 69, 95, 80, 69, 78, 68, 73, 78, 71]),
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
    ".pdf",
    ".ipynb",
}
IGNORED = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "build", "dist"}
FORBIDDEN_DIRS = {"docs", "data", "raw", "external", "extracted", "derived", "results", "outputs", "logs"}
FORBIDDEN_FILES = {"PUBLIC_RELEASE_MANIFEST.tsv", "RELEASE_BUILD_REPORT.md"}
STUB_PATTERNS = [
    re.compile(r"TO" + r"DO"),
    re.compile(r"FIX" + r"ME"),
    re.compile(r"Not" + r"Implemented" + r"Error"),
    re.compile(r"(?m)^\s*pass\s*(#.*)?$"),
]


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in IGNORED or part.endswith((".egg-info", ".dist-info")) for part in rel.parts):
            continue
        yield path


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors: list[str] = []
    for path in iter_files(root):
        rel = path.relative_to(root)
        if any(part in FORBIDDEN_DIRS for part in rel.parts):
            errors.append(f"forbidden directory entry: {rel}")
        if path.is_file() and path.name in FORBIDDEN_FILES:
            errors.append(f"forbidden public artifact: {rel}")
        if path.is_file() and path.suffix.lower() in DATA_EXTENSIONS:
            errors.append(f"forbidden data-like extension: {rel}")
        if path.is_symlink():
            try:
                path.resolve().relative_to(root.resolve())
            except ValueError:
                errors.append(f"symlink points outside folder: {rel}")
            continue
        if not path.is_file():
            continue
        text = read_text(path)
        if text is None:
            continue
        for pattern in FORBIDDEN_TEXT:
            if pattern in str(rel) or pattern in text:
                errors.append(f"forbidden private/internal string found in {rel}")
                break
        if rel.parts[:2] == ("src", "samo_copco") and path.suffix == ".py":
            for pattern in STUB_PATTERNS:
                if pattern.search(text):
                    errors.append(f"stub marker found in {rel}")
                    break
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("public release sanitizer passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
