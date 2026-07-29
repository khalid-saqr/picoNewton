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


def validate_step7(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    gate_path = root / "step7_gate.json"
    manifest_path = root / "step7_manifest.json"
    if not gate_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Step 8 requires Step 7 gate and manifest")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise RuntimeError("Step 8 requires a passing Step 7 gate")
    if manifest.get("status") != "complete" or manifest.get("allowed_next_step") != 8:
        raise RuntimeError("Step 7 manifest does not authorise Step 8")
    for name, record in manifest.get("files", {}).items():
        path = root / name
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise RuntimeError(f"Step 7 checksum failed: {name}")
    required = ["crossed_susceptibility.csv", "step7_archive.npz"]
    for name in required:
        if not (root / name).is_file():
            raise RuntimeError(f"Step 7 output missing: {name}")
    return {"passed": True, "root": str(root), "gate": gate, "manifest": manifest}


def kernel_key(archive: Any, matrix_type: str, vessel_id: str) -> str:
    candidates = (
        f"{matrix_type}__{vessel_id}__kernel2",
        f"{matrix_type}__{vessel_id}__K2",
    )
    for candidate in candidates:
        if candidate in archive:
            return candidate
    raise KeyError(f"Step 7 archive lacks a second-order kernel for {matrix_type}/{vessel_id}")


def summarise_errors(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(group_columns, dropna=False)["relative_error"]
        .agg(
            count="size",
            median_relative_error="median",
            mean_relative_error="mean",
            p90_relative_error=lambda values: float(values.quantile(0.90)),
            maximum_relative_error="max",
        )
        .reset_index()
    )


def file_records(root: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    records = {}
    for name in names:
        path = root / name
        records[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return records
