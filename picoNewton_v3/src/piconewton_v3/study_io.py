from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "1.0.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deterministic_run_id(config: dict[str, Any], code_commit: str) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "code_commit": code_commit,
    }
    return "run_" + sha256_bytes(canonical_json(payload).encode("utf-8"))[:16]


def safe_relative_path(relative: str) -> Path:
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    return Path(*candidate.parts)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def mount_google_drive(mount_point: str = "/content/drive") -> Path:
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("Google Drive mounting is available only in Colab.") from exc
    drive.mount(mount_point, force_remount=False)
    root = Path(mount_point)
    if not root.exists():
        raise RuntimeError("Google Drive mount did not create the expected path.")
    return root


def resolve_study_root(
    mode: str = "auto",
    drive_subdir: str = "MyDrive/picoNewton_v3",
    local_root: str | Path = "./picoNewton_v3",
) -> tuple[Path, str]:
    env_root = os.environ.get("PICONEWTON_V3_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve(), "environment"

    if mode not in {"auto", "drive", "local"}:
        raise ValueError("mode must be auto, drive, or local")

    in_colab = "google.colab" in sys.modules
    if mode == "drive" or (mode == "auto" and in_colab):
        drive_root = mount_google_drive()
        return (drive_root / safe_relative_path(drive_subdir)).resolve(), "drive"

    return Path(local_root).expanduser().resolve(), "local"


class StudyStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative: str) -> Path:
        return self.root / safe_relative_path(relative)

    def initialize_layout(self) -> None:
        for folder in [
            "inputs/raw",
            "inputs/curated",
            "inputs/manifests",
            "configs",
            "runs",
            "publication_bundle",
            "cache/external",
        ]:
            self.path(folder).mkdir(parents=True, exist_ok=True)

    def write_json(self, relative: str, data: Any) -> Path:
        path = self.path(relative)
        atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True))
        return path

    def write_csv(self, relative: str, frame: pd.DataFrame) -> Path:
        path = self.path(relative)
        atomic_write_text(path, frame.to_csv(index=False))
        return path

    def write_npz(self, relative: str, **arrays: np.ndarray) -> Path:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".npz", dir=path.parent, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            np.savez_compressed(tmp_path, **arrays)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return path

    def create_run(
        self,
        config: dict[str, Any],
        code_commit: str,
        v2_blob_sha: str,
        solver_mode: str,
        random_seed: int,
    ) -> tuple[str, Path]:
        if solver_mode not in {"reproduction", "verified"}:
            raise ValueError("solver_mode must be reproduction or verified")
        run_id = deterministic_run_id(config, code_commit)
        run_root = self.path(f"runs/{run_id}")
        for subfolder in [
            "logs",
            "checkpoints",
            "fields",
            "spectra",
            "summaries",
            "figures",
            "provenance",
        ]:
            (run_root / subfolder).mkdir(parents=True, exist_ok=True)

        manifest_path = run_root / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["config_sha256"] != sha256_bytes(
                canonical_json(config).encode()
            ):
                raise RuntimeError("Existing run ID has a different configuration.")
            return run_id, run_root

        manifest = {
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "status": "initialized",
            "config_sha256": sha256_bytes(canonical_json(config).encode()),
            "code_commit": code_commit,
            "v2_blob_sha": v2_blob_sha,
            "solver_mode": solver_mode,
            "random_seed": int(random_seed),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "input_files": [],
            "output_files": [],
        }
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
        self.write_json(f"configs/{run_id}.json", config)
        return run_id, run_root

    def register_file(self, run_id: str, relative_to_run: str, role: str) -> None:
        run_root = self.path(f"runs/{run_id}")
        target = run_root / safe_relative_path(relative_to_run)
        if not target.is_file():
            raise FileNotFoundError(target)
        manifest_path = run_root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        record = {
            "path": relative_to_run,
            "role": role,
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
        entries = manifest["input_files"] if role == "input" else manifest["output_files"]
        entries[:] = [item for item in entries if item["path"] != relative_to_run]
        entries.append(record)
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))

    def set_status(self, run_id: str, status: str) -> None:
        allowed = {"initialized", "running", "complete", "failed"}
        if status not in allowed:
            raise ValueError(status)
        manifest_path = self.path(f"runs/{run_id}/run_manifest.json")
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = status
        manifest["updated_utc"] = utc_now()
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))

    def write_checksums(self, relative_root: str, output_relative: str) -> Path:
        root = self.path(relative_root)
        records = []
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.name == Path(output_relative).name:
                continue
            records.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
        output = root / safe_relative_path(output_relative)
        atomic_write_text(output, "\n".join(records) + "\n")
        return output

    def write_hdf5(self, relative: str, groups: dict[str, dict[str, np.ndarray]]) -> Path:
        """Write a compact HDF5 dataset with explicit group/dataset names."""
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("h5py is required for HDF5 export") from exc
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".h5", dir=path.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with h5py.File(tmp_path, "w") as handle:
                handle.attrs["schema_version"] = SCHEMA_VERSION
                for group_name, datasets in groups.items():
                    group = handle.create_group(group_name)
                    for dataset_name, value in datasets.items():
                        group.create_dataset(dataset_name, data=np.asarray(value), compression="gzip")
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return path
