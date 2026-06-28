#!/usr/bin/env sh
set -eu
out="${1:-/tmp/samo_example}"
python -m samo_copco.cli run-synthetic --out "$out"
