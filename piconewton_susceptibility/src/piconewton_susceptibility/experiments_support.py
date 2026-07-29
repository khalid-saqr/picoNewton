from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from piconewton_v3 import V2_ARTERY_CASES


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_step6(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    gate_path = root / "step6_gate.json"
    manifest_path = root / "step6_manifest.json"
    if not gate_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Step 7 requires the Step 6 gate and manifest")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise RuntimeError("Step 7 requires a passing Step 6 gate")
    if manifest.get("status") != "complete" or manifest.get("allowed_next_step") != 7:
        raise RuntimeError("Step 6 manifest does not authorize Step 7")
    for name, record in manifest.get("files", {}).items():
        path = root / name
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise RuntimeError(f"Step 6 artifact failed checksum validation: {name}")
    return {"passed": True, "root": str(root), "gate": gate, "manifest": manifest}


def matrix_effect_rows(matrix_type: str, values: np.ndarray) -> list[dict[str, Any]]:
    grand = float(np.mean(values))
    rows: list[dict[str, Any]] = []
    for case, mean_value in zip(V2_ARTERY_CASES, np.mean(values, axis=1), strict=True):
        rows.append(
            {
                "matrix_type": matrix_type,
                "effect_type": "vessel",
                "level_id": case.artery_id,
                "mean_phi_rms": float(mean_value),
                "log_effect": float(np.log(mean_value) - np.log(grand)),
            }
        )
    for case, mean_value in zip(V2_ARTERY_CASES, np.mean(values, axis=0), strict=True):
        rows.append(
            {
                "matrix_type": matrix_type,
                "effect_type": "waveform",
                "level_id": case.artery_id,
                "mean_phi_rms": float(mean_value),
                "log_effect": float(np.log(mean_value) - np.log(grand)),
            }
        )
    return rows
