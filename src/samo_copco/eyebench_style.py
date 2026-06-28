"""Non-official EyeBench-style diagnostics for SAMO outputs."""

from __future__ import annotations

import json
from pathlib import Path


def write_nonofficial_diagnostic(metrics_path: str | Path, output_path: str | Path) -> dict[str, object]:
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    report = {
        "diagnostic_name": "non-official EyeBench-style diagnostics",
        "official_status": False,
        "reader_disjoint": True,
        "n_readers": metrics.get("n_readers"),
        "roc_auc": metrics.get("roc_auc"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
    }
    text = "\n".join(
        [
            "# Non-official EyeBench-style Diagnostics",
            "",
            "This diagnostic is not an official EyeBench result.",
            "It reports reader-disjoint SAMO metrics for local analysis only.",
            "",
            f"- Readers: {report['n_readers']}",
            f"- ROC AUC: {report['roc_auc']}",
            f"- Balanced accuracy: {report['balanced_accuracy']}",
        ]
    )
    Path(output_path).write_text(text + "\n", encoding="utf-8")
    return report
