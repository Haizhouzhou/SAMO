"""Command line interface for SAMO."""

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
from .lm_scoring import DEFAULT_DFM_MODEL_ID, LMScoringConfig, score_word_table, validate_lm_feature_table, write_lm_feature_outputs
from .profiles import build_reader_profiles
from .residualization import crossfit_residualize_by_reader
from .validation import validate_configured_data, validate_profile_table


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def validate_data_command(args: argparse.Namespace) -> int:
    print(validate_configured_data(args.config, dry_run=args.dry_run))
    return 0


def prepare_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    word, labels = load_configured_inputs(config)
    prepared = prepare_word_table(word, labels)
    output = Path(args.out or config.get("paths", {}).get("output_dir", "samo_outputs")) / "prepared_word_features.csv"
    write_table(prepared, output)
    print(output)
    return 0


def score_predictability_command(args: argparse.Namespace) -> int:
    if args.config:
        config_payload = load_config(args.config)
        lm_cfg = config_payload.get("language_models", {})
        primary = lm_cfg.get("primary_surprisal", {})
        args.model_id = args.model_id or primary.get("model_id", DEFAULT_DFM_MODEL_ID)
        args.tokenizer_id = args.tokenizer_id or primary.get("tokenizer_id")
        args.context_mode = args.context_mode or lm_cfg.get("context_mode", "paragraph")
    if args.input is None:
        raise ValueError("--input is required")
    output_path = args.out or "/tmp/samo_lm_features.csv"
    frame = read_table(args.input)
    config = LMScoringConfig(
        model_id=args.model_id or DEFAULT_DFM_MODEL_ID,
        tokenizer_id=args.tokenizer_id,
        context_mode=args.context_mode,
        device=args.device,
        dtype=args.dtype,
        mock_model=bool(args.mock_model),
        dry_run=bool(args.dry_run),
        shard_id=int(args.shard_id),
        num_shards=int(args.num_shards),
    )
    if args.mock_model and args.real_run:
        raise ValueError("--mock-model and --real-run cannot be combined")
    result = score_word_table(frame, config)
    if args.dry_run:
        _print_json(result.manifest)
        return 0
    validate_lm_feature_table(result.lm_features).raise_for_errors()
    output, manifest, alignment = write_lm_feature_outputs(
        result,
        output_path,
        manifest_path=args.manifest,
        alignment_report_path=args.alignment_report,
    )
    _print_json({"output": str(output), "manifest": str(manifest), "alignment_report": str(alignment)})
    return 0


def build_profiles_command(args: argparse.Namespace) -> int:
    frame = prepare_word_table(read_table(args.input))
    transformed = crossfit_residualize_by_reader(frame)
    profiles = build_reader_profiles(transformed)
    validate_profile_table(profiles).raise_for_errors()
    write_table(profiles, args.out)
    print(args.out)
    return 0


def evaluate_lopo_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    word, labels = load_configured_inputs(config)
    result = run_lopo_evaluation(word, labels)
    output_dir = Path(args.out or config.get("paths", {}).get("output_dir", "samo_lopo_outputs"))
    write_lopo_outputs(result, output_dir)
    print(output_dir)
    return 0


def ablate_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    word, labels = load_configured_inputs(config)
    rows = run_ablation_suite(word, labels)
    write_table(rows, args.out)
    print(args.out)
    return 0


def eyebench_nonofficial_command(args: argparse.Namespace) -> int:
    report = write_nonofficial_diagnostic(args.metrics, args.out)
    _print_json(report)
    return 0


def make_tables_figures_command(args: argparse.Namespace) -> int:
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = [{"metric": key, "value": value} for key, value in sorted(metrics.items())]
    write_table(__import__("pandas").DataFrame(rows), args.out)
    print(args.out)
    return 0


def run_synthetic_command(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    word_path, label_path = synthetic_fixture_paths(repo_root)
    repeated_words = read_table(word_path)
    labels = read_table(label_path)
    lm_result = score_word_table(repeated_words, LMScoringConfig(mock_model=True, context_mode="paragraph"))
    result = run_lopo_evaluation(lm_result.frame, labels)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(lm_result.frame, output_dir / "synthetic_word_features_with_lm.csv")
    write_lopo_outputs(result, output_dir)
    summary = {
        "status": "complete",
        "n_readers": int(result["metrics"]["n_readers"]),
        "metrics_path": str(output_dir / "metrics.json"),
        "profiles_path": str(output_dir / "reader_profiles.csv"),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_json(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="samo-copco", description="SAMO public commands")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=False)

    validate = sub.add_parser("validate-data", help="validate configured local CopCo inputs")
    validate.add_argument("--config", required=True)
    validate.add_argument("--dry-run", action="store_true")
    validate.set_defaults(func=validate_data_command)

    prepare = sub.add_parser("prepare", help="prepare normalized SAMO word rows")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--out")
    prepare.set_defaults(func=prepare_command)

    score = sub.add_parser("score-predictability", help="score word-level DFM causal-LM surprisal and entropy")
    score.add_argument("--input", required=True)
    score.add_argument("--out")
    score.add_argument("--config")
    score.add_argument("--model-id", default=DEFAULT_DFM_MODEL_ID)
    score.add_argument("--tokenizer-id")
    score.add_argument("--context-mode", default="paragraph", choices=["paragraph", "sentence", "text"])
    score.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    score.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    score.add_argument("--mock-model", action="store_true")
    score.add_argument("--real-run", action="store_true")
    score.add_argument("--dry-run", action="store_true")
    score.add_argument("--shard-id", type=int, default=0)
    score.add_argument("--num-shards", type=int, default=1)
    score.add_argument("--manifest")
    score.add_argument("--alignment-report")
    score.set_defaults(func=score_predictability_command)

    profiles = sub.add_parser("build-profiles", help="build one SAMO reader profile per reader")
    profiles.add_argument("--input", required=True)
    profiles.add_argument("--out", required=True)
    profiles.set_defaults(func=build_profiles_command)

    lopo = sub.add_parser("evaluate-lopo", help="run reader-disjoint LOPO evaluation")
    lopo.add_argument("--config", required=True)
    lopo.add_argument("--out")
    lopo.set_defaults(func=evaluate_lopo_command)

    ablate = sub.add_parser("ablate", help="run LM exposure and sensitivity ablations")
    ablate.add_argument("--config", required=True)
    ablate.add_argument("--out", required=True)
    ablate.set_defaults(func=ablate_command)

    eye = sub.add_parser("eyebench-style-nonofficial", help="write non-official EyeBench-style diagnostics")
    eye.add_argument("--metrics", required=True)
    eye.add_argument("--out", required=True)
    eye.set_defaults(func=eyebench_nonofficial_command)

    tables = sub.add_parser("make-tables-figures", help="make compact metric tables")
    tables.add_argument("--metrics", required=True)
    tables.add_argument("--out", required=True)
    tables.set_defaults(func=make_tables_figures_command)

    synth = sub.add_parser("run-synthetic", help="run the synthetic SAMO pipeline")
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
