from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_step8(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    gate_path = root / "step8_gate.json"
    manifest_path = root / "step8_manifest.json"
    if not gate_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Step 9 requires Step 8 gate and manifest")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise RuntimeError("Step 9 requires a passing Step 8 gate")
    if manifest.get("status") != "complete" or manifest.get("allowed_next_step") != 9:
        raise RuntimeError("Step 8 manifest does not authorise Step 9")
    for name, record in manifest.get("files", {}).items():
        path = root / name
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise RuntimeError(f"Step 8 checksum failed: {name}")
    required = ("reduced_law.json", "step8_reduced_law.npz")
    for name in required:
        if not (root / name).is_file():
            raise RuntimeError(f"Step 8 output missing: {name}")
    return {"passed": True, "root": str(root), "gate": gate, "manifest": manifest}


def file_records(root: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": sha256(root / name), "bytes": (root / name).stat().st_size}
        for name in names
    }


def error_summary(values: pd.Series | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def frozen_law(root: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    law = json.loads((root / "reduced_law.json").read_text(encoding="utf-8"))
    archive = np.load(root / "step8_reduced_law.npz")
    selected_kernel = np.asarray(archive["selected_kernel"], dtype=complex)
    parameters = np.asarray(archive["scale_parameters"], dtype=float)
    if law.get("selected_rank") != 1 or selected_kernel.shape != (12, 12):
        raise RuntimeError("Step 9 requires the frozen rank-one Step 8 law")
    if parameters.shape != (3,):
        raise RuntimeError("Step 8 vessel-scale parameter vector is malformed")
    return law, selected_kernel, parameters
