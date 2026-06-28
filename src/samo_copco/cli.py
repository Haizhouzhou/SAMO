"""Command line interface for the SAMO public release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import __version__
from .ablations import run_ablation_suite
from .data import load_config, load_configured_inputs, prepare_word_table, read_table, synthetic_fixture_paths, write_table
from .evaluation import run_lopo_evaluation, write_lopo_outputs
from .eyebench_style import write_nonofficial_diagnostic
from .predictability import add_predictability_columns
from .profiles import build_reader_profiles
from .residualization import FoldLocalResidualizer
from .validation import validate_configured_data


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_prepared_from_config(config_path: str | Path):
    config = load_config(config_path)
    word, labels = load_configured_inputs(config)
    return prepare_word_table(word, labels), labels, config


def validate_data_command(args: argparse.Namespace) -> int:
    message = validate_configured_data(args.config, dry_run=args.dry_run)
    print(message)
    return 0


def prepare_features_command(args: argparse.Namespace) -> int:
    prepared, _labels, config = _load_prepared_from_config(args.config)
    output_dir = Path(config.get("paths", {}).get("output_dir", args.out or "samo_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(prepared, output_dir / "prepared_word_features.csv")
    print(output_dir / "prepared_word_features.csv")
    return 0


def score_predictability_command(args: argparse.Namespace) -> int:
    frame = read_table(args.input)
    scored = add_predictability_columns(frame)
    write_table(scored, args.out)
    print(args.out)
    return 0


def build_profiles_command(args: argparse.Namespace) -> int:
    frame = read_table(args.input)
    prepared = prepare_word_table(frame)
    residualizer = FoldLocalResidualizer().fit(prepared)
    transformed = residualizer.transform(prepared)
    profiles = build_reader_profiles(transformed)
    write_table(profiles, args.out)
    print(args.out)
    return 0


def run_lopo_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    word, labels = load_configured_inputs(config)
    result = run_lopo_evaluation(word, labels)
    output_dir = Path(args.out or config.get("paths", {}).get("output_dir", "samo_lopo_outputs"))
    write_lopo_outputs(result, output_dir)
    print(output_dir)
    return 0


def run_ablations_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    word, labels = load_configured_inputs(config)
    rows = run_ablation_suite(word, labels)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False)
    print(output)
    return 0


def run_nonofficial_command(args: argparse.Namespace) -> int:
    report = write_nonofficial_diagnostic(args.metrics, args.out)
    print(json.dumps(report, sort_keys=True))
    return 0


def make_tables_figures_command(args: argparse.Namespace) -> int:
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["metric,value"] + [f"{key},{value}" for key, value in sorted(metrics.items())]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 0


def run_synthetic_command(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    word_path, label_path = synthetic_fixture_paths(repo_root)
    word = read_table(word_path)
    labels = read_table(label_path)
    prepared = prepare_word_table(word, labels)
    result = run_lopo_evaluation(prepared)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(prepared, output_dir / "prepared_word_features.csv")
    write_lopo_outputs(result, output_dir)
    summary = {
        "status": "complete",
        "n_readers": int(result["metrics"]["n_readers"]),
        "metrics_path": str(output_dir / "metrics.json"),
        "profiles_path": str(output_dir / "reader_profiles.csv"),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="samo-copco", description="SAMO public release commands")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=False)

    validate = sub.add_parser("validate-data", help="validate configured local CopCo inputs")
    validate.add_argument("--config", required=True)
    validate.add_argument("--dry-run", action="store_true")
    validate.set_defaults(func=validate_data_command)

    prep = sub.add_parser("prepare-features", help="prepare a public SAMO word table from configured local files")
    prep.add_argument("--config", required=True)
    prep.add_argument("--out")
    prep.set_defaults(func=prepare_features_command)

    score = sub.add_parser("score-predictability", help="add transparent predictability-style columns")
    score.add_argument("--input", required=True)
    score.add_argument("--out", required=True)
    score.set_defaults(func=score_predictability_command)

    profiles = sub.add_parser("build-profiles", help="build one reader-profile row per reader")
    profiles.add_argument("--input", required=True)
    profiles.add_argument("--out", required=True)
    profiles.set_defaults(func=build_profiles_command)

    lopo = sub.add_parser("run-lopo", help="run reader-disjoint LOPO evaluation")
    lopo.add_argument("--config", required=True)
    lopo.add_argument("--out")
    lopo.set_defaults(func=run_lopo_command)

    ablations = sub.add_parser("run-ablations", help="run predefined SAMO ablations")
    ablations.add_argument("--config", required=True)
    ablations.add_argument("--out", required=True)
    ablations.set_defaults(func=run_ablations_command)

    nonofficial = sub.add_parser("run-eyebench-style-nonofficial", help="write non-official EyeBench-style diagnostics")
    nonofficial.add_argument("--metrics", required=True)
    nonofficial.add_argument("--out", required=True)
    nonofficial.set_defaults(func=run_nonofficial_command)

    tables = sub.add_parser("make-tables-figures", help="make small metric tables from SAMO outputs")
    tables.add_argument("--metrics", required=True)
    tables.add_argument("--out", required=True)
    tables.set_defaults(func=make_tables_figures_command)

    synth = sub.add_parser("run-synthetic", help="run a real synthetic end-to-end SAMO pipeline")
    synth.add_argument("--out", required=True)
    synth.set_defaults(func=run_synthetic_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
