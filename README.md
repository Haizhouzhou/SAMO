# SAMO for CopCo Reader Profiles

SAMO is a small public release candidate for residualized predictability-sensitive reader-profile modeling. The public task is CopCo dyslexia-labelled vs typical/control reader-level classification.

This repository contains code, documentation, tests, and synthetic fixtures only. It does not redistribute CopCo data, participant-level prediction tables, word-level real-data feature tables, model checkpoints, or local analysis artifacts. Users must obtain CopCo through the appropriate data access channel and provide local paths in configuration files.

## Install

```bash
python -m pip install -e '.[dev]'
```

## Public Commands

```bash
python -m samo_copco.cli --help
python -m samo_copco.cli validate-data --config configs/copco_paths.example.yaml --dry-run
python -m samo_copco.cli run-synthetic --out /tmp/samo_synthetic
```

The synthetic run exercises the same public modules used by the real-data commands: schema validation, word-table preparation, predictability-like columns, fold-local residualization, reader-profile aggregation, reader-disjoint LOPO evaluation, and metric writing.

## Claim Boundaries

The target label is reader-level. Word-level observations are repeated evidence and are not independent target samples. Main evaluation uses reader-disjoint LOPO logic. Residualizers are fit only on training readers within each fold. Direct reader IDs, labels, targets, speech IDs, and text IDs are excluded from predictors.

EyeBench-related material in this release is non-official EyeBench-style diagnostic code only. It is not an official leaderboard submission or an official benchmark result.

This repository does not support clinical diagnosis, screening readiness, deployment readiness, medical utility, or external-generalization claims.

## License

Repository code license: `MIT`. CopCo data are not redistributed by this repository; users must obtain CopCo separately and follow the applicable data access terms.
