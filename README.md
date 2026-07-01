# SAMO

SAMO builds residualized, predictability-sensitive reader profiles for CopCo natural-reading data. The main public workflow scores words with a Danish causal language model, merges word-level surprisal and entropy into repeated reader-word gaze rows, residualizes gaze outcomes inside reader-disjoint folds, and evaluates dyslexia-labelled vs typical/control reader-level classification.

## Repository Layout

- `src/samo_copco/`: package code for data contracts, stimulus reconstruction, language-model scoring, residualization, profiles, LOPO evaluation, ablations, and the command line interface.
- `configs/`: example YAML files for local CopCo paths, DFM scoring, LOPO evaluation, and ablation groups.
- `scripts/`: thin command wrappers around the package CLI.
- `tests/`: synthetic fixtures and tests that exercise the public modules without real CopCo data or model downloads.
- `examples/synthetic_minimal/`: a minimal end-to-end synthetic run.

## Installation

Base installation is lightweight:

```bash
python -m pip install -e '.[dev]'
```

Real language-model scoring needs the optional LM extras:

```bash
python -m pip install -e '.[dev,lm]'
```

The base package imports without PyTorch or Transformers installed. Those packages are imported only when real causal-LM scoring is requested.

## Quick Checks

```bash
python -m samo_copco.cli --help
python -m samo_copco.cli score-predictability --help
python -m samo_copco.cli validate-data --config configs/copco_paths.example.yaml --dry-run
python -m samo_copco.cli run-synthetic --out /tmp/samo_synthetic
```

The synthetic run validates schemas, performs deterministic mock LM scoring, residualizes gaze rows, builds reader profiles, runs reader-disjoint LOPO evaluation, and writes metrics plus profiles.

## CopCo Inputs

CopCo can be obtained from OSF: https://osf.io/ud8s5/

The original CopCo processing repository is: https://github.com/norahollenstein/copco-processing

This repository does not bundle CopCo data. Provide local word-level gaze tables and reader labels through `configs/copco_paths.example.yaml` or equivalent command-line paths.

Word-level inputs may use CopCo camelCase columns or normalized snake_case columns. Accepted aliases include `speechId` to `speech_id`, `paragraphId` to `paragraph_id`, `sentenceId` to `sentence_id`, `wordId` to `source_word_id`, and reader identifiers such as `participantId`, `participant_id`, `subject_id`, or `reader_id`.

CopCo `wordId` values are local to their paragraph or trial. SAMO therefore constructs `lm_stable_word_id` from structured stimulus columns:

```text
speech_id::paragraph_id::sentence_id::source_word_id
```

If `sentence_id` is unavailable, the stable key is:

```text
speech_id::paragraph_id::source_word_id
```

A pre-existing globally unique `word_id` is accepted only for synthetic or simple inputs.

## Language-Model Predictability

The SAMO language-model path uses:

```text
danish-foundation-models/dfm-decoder-open-v0-7b-pt
```

This is a base Danish causal LM used for next-token likelihood. For tokens `x_0 ... x_n`, logits at position `t` predict `x_{t+1}`. Word surprisal is computed from shifted next-token negative log probability of the observed token under its left context. Entropy is computed from the same next-token probability distribution as `-sum p log p`. Natural log units are used by default.

The tokenizer produces subword tokens. SAMO reconstructs paragraph, sentence, or text contexts from the word table, obtains tokenizer offset mappings, assigns each non-special token to the word span with maximum character overlap, and aggregates token scores back to words. Word surprisal is the sum of scored subword surprisals. Word entropy is the mean entropy over scored subwords, and onset entropy is taken from the first scored subword aligned to the word.

Expected LM input columns:

```text
word
speech_id / speechId
paragraph_id / paragraphId
sentence_id / sentenceId
source_word_id / wordId
```

Required LM output columns:

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

Real scoring:

```bash
python -m samo_copco.cli score-predictability \
  --input local_words.csv \
  --out local_lm_features.csv \
  --real-run \
  --context-mode paragraph
```

Mock scoring for tests, with no model download:

```bash
python -m samo_copco.cli score-predictability \
  --input tests/fixtures/synthetic_lm_words.csv \
  --out /tmp/samo_mock_lm_features.csv \
  --mock-model
```

`score-predictability` never creates lexical predictability columns implicitly. A synthetic lexical fallback is available only through explicit synthetic code paths.

## Main Commands

```bash
python -m samo_copco.cli validate-data --config configs/copco_paths.example.yaml --dry-run
python -m samo_copco.cli prepare --word-features words.csv --labels labels.csv --out prepared_words.csv
python -m samo_copco.cli score-predictability --input stimulus_words.csv --out lm_features.csv --real-run
python -m samo_copco.cli build-profiles --input prepared_with_lm.csv --out reader_profiles.csv
python -m samo_copco.cli evaluate-lopo --input prepared_with_lm.csv --labels labels.csv --out results_dir
python -m samo_copco.cli ablate --input prepared_with_lm.csv --labels labels.csv --out ablations_dir
python -m samo_copco.cli eyebench-style-nonofficial --predictions predictions.csv --out summary.json
```

## Reader Profiles

SAMO treats word rows as repeated evidence for reader-profile construction and the label as reader-level. Gaze outcomes are residualized with residualizers fit only on training readers and then applied to held-out readers. Direct identifiers and labels are excluded from predictors.

When LM columns are present, reader profiles contain separate exposure and sensitivity features:

```text
lm_surprisal_exposure_mean
lm_surprisal_exposure_std
lm_entropy_exposure_mean
lm_entropy_exposure_std
sensitivity__<gaze>__surprisal
sensitivity__<gaze>__entropy
```

Sensitivity is the per-reader mean of residual gaze multiplied by `lm_word_surprisal` or by `lm_word_entropy`.

## Exposure And Sensitivity Groups

The ablation selector exposes these groups:

- `lm_exposure_only`: surprisal and entropy exposure summaries.
- `lm_sensitivity_only`: residual-gaze sensitivity to surprisal and entropy.
- `lm_residualized_profile`: residual summaries plus LM sensitivity fields.
- `lm_exposure_plus_sensitivity`: LM exposure and LM sensitivity fields.

## Results Snapshot

Reported SAMO reader-profile results:

- Full reader profile AUROC: 0.895
- Balanced accuracy: 0.842
- PR-AUC: 0.864
- Brier score: 0.116
- Text exposure AUROC: 0.424
- Reader sensitivity AUROC: 0.889
- Residualized profile AUROC: 0.895
- Exposure + sensitivity AUROC: 0.873

## EyeBench Context

SAMO is related to EyeBench-style reader-level evaluation, but this repository does not claim an official EyeBench leaderboard submission. Official EyeBench documentation is available at https://eyebench.github.io/ and the EyeBench code is at https://github.com/EyeBench/eyebench

## Citation

Repository target: https://github.com/Haizhouzhou/SAMO

If you use this code, cite the SAMO repository and cite CopCo separately according to the CopCo data source requirements.

## License

Code in this repository is released under the MIT License.
