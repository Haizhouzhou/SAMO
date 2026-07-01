# SAMO

**SAMO** implements residualized predictability-sensitive reader-profile modeling for natural-reading eye-tracking data.

The main workflow in this repository targets **CopCo dyslexia-labelled vs typical/control reader-level classification**. SAMO represents each reader by how their gaze costs vary with language-model predictability, especially word-level surprisal and entropy from a Danish causal language model.

## Overview

SAMO turns repeated word-level gaze observations into one reader-level profile.

The pipeline is:

1. validate and normalize CopCo-style word-level gaze tables
2. reconstruct Danish stimulus contexts from word rows
3. compute word-level language-model surprisal and entropy
4. align tokenizer subword scores back to CopCo word IDs
5. merge stimulus-level LM features into repeated reader-word gaze rows
6. residualize gaze measures inside reader-disjoint folds
7. aggregate residual gaze responses into reader profiles
8. evaluate reader-level prediction with LOPO splits
9. compare exposure-only and sensitivity-based feature groups

The target is a **reader-level label**. Word-level observations are repeated evidence used to construct reader profiles; they are not independent classification targets.

## Repository Layout

```text
configs/              Example configuration files
examples/             Minimal synthetic example
scripts/              Thin command wrappers
src/samo_copco/       Python package
tests/                Unit tests and synthetic fixtures
```

The repository contains code, configuration templates, tests, and synthetic fixtures. It does not include CopCo data, EyeBench data, raw gaze files, real word-level feature tables, participant-level prediction tables, model checkpoints, or local analysis artifacts.

## Installation

Install the base package with development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Real DFM language-model scoring uses optional dependencies:

```bash
python -m pip install -e '.[dev,lm]'
```

The base package imports without PyTorch or Transformers. Those packages are loaded only when real causal-LM scoring is requested.

## Quick Checks

```bash
python -m samo_copco.cli --help
python -m samo_copco.cli score-predictability --help
python -m samo_copco.cli validate-data --config configs/copco_paths.example.yaml --dry-run
python -m samo_copco.cli run-synthetic --out /tmp/samo_synthetic
pytest -q
```

The synthetic run exercises the same public modules used by real-data commands: schema validation, stimulus reconstruction, mock LM scoring, fold-local residualization, reader-profile aggregation, LOPO evaluation, and metric writing.

## Data

SAMO is designed for **CopCo: The Copenhagen Corpus of Eye-Tracking Recordings from Natural Reading**.

CopCo OSF page:

```text
https://osf.io/ud8s5/
```

CopCo processing code from the original corpus release:

```text
https://github.com/norahollenstein/copco-processing
```

Users should obtain CopCo separately and provide local paths through a copied configuration file:

```bash
cp configs/copco_paths.example.yaml configs/copco_paths.local.yaml
```

Then edit `configs/copco_paths.local.yaml` to point to local files.

Expected local inputs are:

```text
word-level gaze feature table
reader-label table
optional precomputed LM feature table
```

Typical CopCo word-level gaze features include fixation count, first fixation duration, mean fixation duration, total fixation duration, first-pass duration, go-past time, landing position, mean saccade duration, and peak saccade velocity.

## CopCo Column Handling

Word-level inputs may use either CopCo-style camelCase columns or normalized snake_case columns.

Accepted aliases include:

```text
speechId       -> speech_id
paragraphId    -> paragraph_id
sentenceId     -> sentence_id
wordId         -> source_word_id
trialId        -> trial_id
participantId  -> reader_id
participant_id -> reader_id
subject_id     -> reader_id
```

CopCo `wordId` values are local to a paragraph or trial. SAMO therefore constructs a stable stimulus key:

```text
speech_id::paragraph_id::sentence_id::source_word_id
```

If `sentence_id` is unavailable, the stable key is:

```text
speech_id::paragraph_id::source_word_id
```

This key is stored as:

```text
lm_stable_word_id
```

Language-model features are computed at the stimulus-word level and can then be merged many-to-one into repeated participant-word gaze rows.

## Language-Model Predictability

The reported SAMO experiments use the Danish causal decoder:

```text
danish-foundation-models/dfm-decoder-open-v0-7b-pt
```

This model is used as a base causal language model for next-token likelihood. For a token sequence:

```text
x_0, x_1, ..., x_n
```

logits at position `t` predict the next observed token `x_{t+1}`.

Token surprisal is computed as shifted next-token negative log probability:

```text
surprisal(x_{t+1}) = -log p(x_{t+1} | x_0, ..., x_t)
```

Entropy is computed from the same next-token distribution:

```text
entropy_t = - sum_v p(v | x_0, ..., x_t) log p(v | x_0, ..., x_t)
```

Natural log units are used by default.

Because the model tokenizer produces subword tokens, SAMO reconstructs paragraph, sentence, or text contexts from the word table, obtains tokenizer offset mappings, assigns each non-special token to the word span with maximum character overlap, and aggregates token-level scores back to words.

Word-level aggregation:

```text
lm_word_surprisal      = sum of scored subword surprisals
lm_word_entropy        = mean entropy over scored subwords
lm_word_entropy_onset  = entropy at the first scored subword aligned to the word
```

Required LM output columns are:

```text
lm_stable_word_id
speech_id
paragraph_id
sentence_id
source_word_id
word
lm_model_id
lm_tokenizer_id
lm_context_mode
lm_word_surprisal
lm_word_entropy
lm_word_entropy_onset
lm_subword_count
lm_scored_subword_count
lm_alignment_status
lm_alignment_warning
lm_alignment_error
```

## Main Commands

The package can be called either as:

```bash
python -m samo_copco.cli ...
```

or, after installation, through the console script:

```bash
samo-copco ...
```

The examples below use `python -m samo_copco.cli`.

### 1. Validate local data paths

```bash
python -m samo_copco.cli validate-data \
  --config configs/copco_paths.local.yaml
```

For checking the example config without requiring real local files:

```bash
python -m samo_copco.cli validate-data \
  --config configs/copco_paths.example.yaml \
  --dry-run
```

### 2. Prepare normalized word rows

```bash
python -m samo_copco.cli prepare \
  --config configs/copco_paths.local.yaml \
  --out samo_outputs
```

This writes:

```text
samo_outputs/prepared_word_features.csv
```

### 3. Score DFM surprisal and entropy

Real DFM scoring:

```bash
python -m samo_copco.cli score-predictability \
  --input samo_outputs/prepared_word_features.csv \
  --out samo_outputs/dfm_lm_features.csv \
  --model-id danish-foundation-models/dfm-decoder-open-v0-7b-pt \
  --context-mode paragraph \
  --real-run
```

Optional sidecar outputs:

```bash
python -m samo_copco.cli score-predictability \
  --input samo_outputs/prepared_word_features.csv \
  --out samo_outputs/dfm_lm_features.csv \
  --manifest samo_outputs/dfm_lm_features.manifest.json \
  --alignment-report samo_outputs/dfm_lm_features.alignment_report.json \
  --context-mode paragraph \
  --real-run
```

Dry-run column and configuration check:

```bash
python -m samo_copco.cli score-predictability \
  --input samo_outputs/prepared_word_features.csv \
  --config configs/lm_scoring.example.yaml \
  --dry-run
```

No-download mock scoring for tests:

```bash
python -m samo_copco.cli score-predictability \
  --input tests/fixtures/synthetic_lm_words.csv \
  --out /tmp/samo_mock_lm_features.csv \
  --mock-model
```

Mock scoring is deterministic and is used only to exercise the scoring, alignment, and output-schema code paths. It is not the scientific LM used in the reported experiments.

### 4. Build reader profiles

Use a prepared table that already contains LM columns:

```bash
python -m samo_copco.cli build-profiles \
  --input samo_outputs/prepared_with_lm.csv \
  --out samo_outputs/reader_profiles.csv
```

### 5. Evaluate reader-level LOPO prediction

Point `configs/copco_paths.local.yaml` to the prepared table with LM columns, then run:

```bash
python -m samo_copco.cli evaluate-lopo \
  --config configs/copco_paths.local.yaml \
  --out samo_outputs/lopo
```

The output directory contains aggregate metrics and reader-profile outputs.

### 6. Run exposure/sensitivity ablations

```bash
python -m samo_copco.cli ablate \
  --config configs/copco_paths.local.yaml \
  --out samo_outputs/ablations.csv
```

### 7. Write EyeBench-style diagnostic summary

```bash
python -m samo_copco.cli eyebench-style-nonofficial \
  --metrics samo_outputs/lopo/metrics.json \
  --out samo_outputs/eyebench_style_summary.json
```

### 8. Run the synthetic end-to-end example

```bash
python -m samo_copco.cli run-synthetic \
  --out /tmp/samo_synthetic
```

## Reader Profiles

SAMO builds one profile row per reader.

For each gaze outcome, residualizers are fit only on training-reader rows and applied to held-out readers. Direct reader IDs, target labels, speech IDs, paragraph IDs, sentence IDs, text IDs, local word IDs, and stable word IDs are excluded from predictors.

When LM columns are present, reader profiles include exposure features:

```text
lm_surprisal_exposure_mean
lm_surprisal_exposure_std
lm_entropy_exposure_mean
lm_entropy_exposure_std
```

and sensitivity features:

```text
sensitivity__<gaze>__surprisal
sensitivity__<gaze>__entropy
```

For a residualized gaze feature `resid__<gaze>`, sensitivity is computed as the reader-level mean interaction between residual gaze and LM predictability:

```text
mean(resid__<gaze> * lm_word_surprisal)
mean(resid__<gaze> * lm_word_entropy)
```

This separates the predictability profile of the material from the reader’s gaze response to that predictability.

## Feature Groups

The ablation helpers define four LM feature groups:

```text
lm_exposure_only
lm_sensitivity_only
lm_residualized_profile
lm_exposure_plus_sensitivity
```

The intended interpretation is:

```text
lm_exposure_only:
  the predictability profile of the material encountered by the reader

lm_sensitivity_only:
  residual gaze response to surprisal and entropy

lm_residualized_profile:
  residual gaze summaries plus LM sensitivity features

lm_exposure_plus_sensitivity:
  exposure summaries plus sensitivity features
```

These selectors exclude direct IDs and target columns.

## Results Snapshot

The associated CopCo analysis reports the following aggregate reader-level results.

Full reader profile on CopCo TYP:

| Evaluation | AUROC | PR-AUC | Balanced accuracy | F1 | Brier |
|---|---:|---:|---:|---:|---:|
| Full reader profile | 0.895 | 0.864 | 0.842 | 0.842 | 0.116 |

Feature ablation:

| Feature family | AUROC | PR-AUC | Balanced accuracy | F1 | Brier |
|---|---:|---:|---:|---:|---:|
| Text exposure | 0.424 | 0.369 | 0.447 | 0.439 | 0.268 |
| Reader sensitivity | 0.889 | 0.861 | 0.842 | 0.842 | 0.113 |
| Residualized profile | 0.895 | 0.864 | 0.842 | 0.842 | 0.116 |
| Exposure + sensitivity | 0.873 | 0.856 | 0.816 | 0.807 | 0.121 |

The ablation separates the predictability profile of the text from the reader’s response to predictability. In the reported results, reader sensitivity accounts for most of the predictive signal.

## EyeBench Context

EyeBench is a benchmark for predictive modeling from eye movements in reading:

```text
https://eyebench.github.io/
https://github.com/EyeBench/eyebench
```

EyeBench evaluates models under regimes such as unseen readers, unseen texts, and jointly unseen readers and texts. This repository includes local EyeBench-style diagnostic utilities for comparing SAMO outputs under related CopCo settings. For official EyeBench submission or leaderboard evaluation, use the official EyeBench package and protocol.

## Development

Run tests:

```bash
pytest -q
```

Run the minimal synthetic example:

```bash
bash examples/synthetic_minimal/run.sh
```

Build a local distribution:

```bash
python -m build
```

Run the public-release sanitizer:

```bash
python scripts/check_public_release.py
```

## Citation

If you use this repository, please cite the SAMO code and the underlying data/benchmark resources.

```bibtex
@misc{samo_copco,
  title = {SAMO: Residualized Predictability-Sensitive Reader-Profile Modeling for CopCo},
  author = {Zheng, Haizhou and Rauss, Sabrina and Fan, Zhaoyan},
  year = {2026},
  url = {https://github.com/Haizhouzhou/SAMO}
}
```

CopCo:

```bibtex
@inproceedings{hollenstein2022copco,
  title = {The Copenhagen Corpus of Eye Tracking Recordings from Natural Reading of Danish Texts},
  author = {Hollenstein, Nora and Barrett, Maria and Björnsdóttir, Marina},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  pages = {1712--1720},
  year = {2022},
  publisher = {European Language Resources Association}
}
```

EyeBench:

```bibtex
@inproceedings{shubi2025eyebench,
  title = {EyeBench: Predictive Modeling from Eye Movements in Reading},
  author = {Shubi, Omer and Reich, David R. and Gruteke Klein, Keren and Angel, Yuval and Prasse, Paul and Jäger, Lena and Berzak, Yevgeni},
  booktitle = {NeurIPS Datasets and Benchmarks},
  year = {2025}
}
```

## License

MIT license for the code in this repository.
