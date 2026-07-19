"""Validation helpers for workflow output directories."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


REQUIRED_OUTPUTS = (
    "six_artery_summary.csv",
    "hypothesis_effects.csv",
    "loao_vector_surrogates.csv",
    "artery_feature_distances.csv",
    "waveforms.npz",
    "validation.json",
    "manifest.json",
)


def validate_output_directory(output_root: Path) -> dict[str, object]:
    output_root = Path(output_root)
    missing = [name for name in REQUIRED_OUTPUTS if not (output_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing workflow outputs: {missing}")

    summary = pd.read_csv(output_root / "six_artery_summary.csv")
    effects = pd.read_csv(output_root / "hypothesis_effects.csv")
    validation = json.loads((output_root / "validation.json").read_text(encoding="utf-8"))
    arrays = np.load(output_root / "waveforms.npz")

    report = {
        "passed": bool(
            summary["artery_id"].nunique() == 6
            and len(effects) > 0
            and len(arrays.files) > 0
            and validation.get("status") == "passed_structural_validation"
        ),
        "arteries": int(summary["artery_id"].nunique()),
        "summary_rows": int(len(summary)),
        "effect_rows": int(len(effects)),
        "waveform_arrays": int(len(arrays.files)),
        "maximum_probability_sum_error": float(
            validation["maximum_probability_sum_error"]
        ),
        "minimum_probability": float(validation["minimum_probability"]),
    }
    return report


__all__ = ["REQUIRED_OUTPUTS", "validate_output_directory"]
