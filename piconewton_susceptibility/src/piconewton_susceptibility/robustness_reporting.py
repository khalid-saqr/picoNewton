from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .robustness_claims import claim_lock
from .robustness_core import Step9Config
from .robustness_metrics import calculate_metrics_and_gates
from .robustness_support import file_records


def close_step9(
    output_root: Path,
    config: Step9Config,
    law: dict[str, Any],
    continuity: dict[str, Any],
    path_frame: pd.DataFrame,
    scale_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    exact_frame: pd.DataFrame,
    eta_frame: pd.DataFrame,
    resolution_frame: pd.DataFrame,
    archive_arrays: dict[str, np.ndarray],
    maximum_residual: float,
) -> dict[str, Any]:
    metric_frame, gates = calculate_metrics_and_gates(
        config,
        continuity,
        path_frame,
        scale_frame,
        prediction_frame,
        exact_frame,
        eta_frame,
        resolution_frame,
        maximum_residual,
    )
    tables = {
        "step8_law_continuity.csv": pd.DataFrame([continuity]),
        "constitutive_path_summary.csv": path_frame,
        "constitutive_path_metrics.csv": metric_frame,
        "constitutive_shape_predictions.csv": prediction_frame,
        "constitutive_scale_ratios.csv": scale_frame,
        "finite_epsilon_closure.csv": exact_frame,
        "eta_robustness.csv": eta_frame,
        "resolution_robustness.csv": resolution_frame,
    }
    for name, frame in tables.items():
        frame.to_csv(output_root / name, index=False)
    np.savez_compressed(output_root / "step9_archive.npz", **archive_arrays)
    (output_root / "step9_gate.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True), encoding="utf-8"
    )
    claim = claim_lock(law, bool(gates["passed"]))
    (output_root / "claim_lock.json").write_text(
        json.dumps(claim, indent=2, sort_keys=True), encoding="utf-8"
    )
    output_names = [*tables, "step9_archive.npz", "step9_gate.json", "claim_lock.json"]
    manifest = {
        "step": 9,
        "status": "complete" if gates["passed"] else "failed",
        "profile": config.profile,
        "scientific_scope": "constitutive_numerical_robustness_and_claim_lock",
        "frozen_step8_law": True,
        "selected_rank": 1,
        "allowed_next_step": 10 if gates["passed"] else None,
        "gates": gates,
        "files": file_records(output_root, output_names),
    }
    (output_root / "step9_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "gates": gates,
        "claim_lock": claim,
        "path_metrics": metric_frame,
    }
