from __future__ import annotations

import pandas as pd

from samo_copco.ablations import ablation_definitions, feature_group_columns


def _profiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reader_id": ["r1", "r2"],
            "reader_label": [0, 1],
            "lm_surprisal_exposure_mean": [1.0, 2.0],
            "lm_entropy_exposure_mean": [3.0, 4.0],
            "lm_surprisal_exposure_std": [0.1, 0.2],
            "lm_entropy_exposure_std": [0.3, 0.4],
            "sensitivity__gaze_duration_ms__surprisal": [0.5, 0.6],
            "sensitivity__gaze_duration_ms__entropy": [0.7, 0.8],
            "residual_mean__gaze_duration_ms": [0.0, 0.0],
            "residual_std__gaze_duration_ms": [1.0, 1.1],
        }
    )


def test_lm_ablation_groups_select_expected_columns_without_ids() -> None:
    profiles = _profiles()
    exposure = feature_group_columns(profiles, "lm_exposure_only")
    sensitivity = feature_group_columns(profiles, "lm_sensitivity_only")
    both = feature_group_columns(profiles, "lm_exposure_plus_sensitivity")
    assert all("exposure" in column for column in exposure)
    assert all(column.startswith("sensitivity__") for column in sensitivity)
    assert set(exposure) < set(both)
    assert set(sensitivity) < set(both)
    for columns in ablation_definitions(profiles).values():
        assert "reader_id" not in columns
        assert "reader_label" not in columns
