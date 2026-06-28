from __future__ import annotations

import sys

from samo_copco.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run-eyebench-style-nonofficial", *sys.argv[1:]]))
