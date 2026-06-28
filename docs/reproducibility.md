# Reproducibility

Install the package in editable mode with development dependencies, then run the synthetic pipeline:

```bash
python -m pip install -e '.[dev]'
python -m samo_copco.cli run-synthetic --out /tmp/samo_synthetic
```

For real local data, copy `configs/copco_paths.example.yaml`, replace placeholders with local files obtained separately, and run validation before analysis:

```bash
python -m samo_copco.cli validate-data --config configs/copco_paths.example.yaml --dry-run
```

A reproducible real-data run should record the input file references, configuration, package version, command line, and output checksums. This release does not claim full reproduction without externally obtained CopCo data.
