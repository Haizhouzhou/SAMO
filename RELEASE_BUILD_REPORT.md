# Release Build Report

This public release candidate packages SAMO for residualized predictability-sensitive reader-profile modeling on the CopCo dyslexia-labelled vs typical/control reader-level classification task.

## Package Layout

The `samo_copco` package provides data contracts, input validation, predictability-style feature preparation, fold-local residualization, reader-profile aggregation, reader-disjoint LOPO evaluation, ablations, metrics, and non-official EyeBench-style diagnostics.

## Data Exclusion

No CopCo data, raw gaze files, participant-level prediction rows, real word-level feature tables, model checkpoints, or local build-audit artifacts are included.

## Benchmark Boundary

EyeBench-related outputs are non-official EyeBench-style diagnostics only.

## Validation Summary

| Command ID | Result | Exit code |
| --- | --- | --- |
| 01 | PASS | 0 |
| 02 | PASS | 0 |
| 03 | PASS | 0 |
| 04 | PASS | 0 |
| 05 | PASS | 0 |
| 06 | PASS | 0 |
| 07 | PASS | 0 |
| 08 | PASS | 1 |
| 09 | PASS | 0 |
| 10 | PASS | 0 |
| 11 | PASS | 0 |
| 12 | PASS | 0 |
| 13 | PASS | 0 |
| 14 | PASS | 0 |

Command 08 is successful only when it exits with code 1 and empty output, because the scanner reports code 1 for no matches.

## Public-Facing Blockers

No public-facing blocker is recorded after the targeted repair pass. Repository code license: `MIT`. CopCo data are not redistributed by this repository.
