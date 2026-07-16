"""Google Colab and local runtime management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import secrets
import shutil
from typing import Mapping


@dataclass(frozen=True)
class RuntimePaths:
    run_id: str
    local_root: Path
    persistent_root: Path
    config: Path
    checkpoints: Path
    figures: Path
    source_data: Path
    logs: Path
    manifests: Path
    validation: Path

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in self.__dict__.items()}


def in_colab() -> bool:
    return "google.colab" in __import__("sys").modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def mount_google_drive(mount_point: str = "/content/drive") -> Path:
    if not in_colab():
        raise RuntimeError("Google Drive mounting is available only in Google Colab.")
    from google.colab import drive  # type: ignore
    drive.mount(mount_point)
    return Path(mount_point)


def create_runtime(*, drive_root: Path, local_base: Path = Path("/content/piconewton_v4_runtime"), prefix: str = "picoNewton_v4", metadata: Mapping[str, object] | None = None) -> RuntimePaths:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{prefix}_{stamp}_{secrets.token_hex(4)}"
    local_root = local_base / run_id
    persistent_root = drive_root / run_id
    if local_root.exists() or persistent_root.exists():
        raise FileExistsError(f"Runtime collision for {run_id}")
    local_root.mkdir(parents=True, exist_ok=False)
    persistent_root.mkdir(parents=True, exist_ok=False)
    children = {}
    for name in ("config", "checkpoints", "figures", "source_data", "logs", "manifests", "validation"):
        path = persistent_root / name
        path.mkdir()
        children[name] = path
    runtime = RuntimePaths(run_id=run_id, local_root=local_root, persistent_root=persistent_root, **children)
    manifest = {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(), "status": "initialized", "paths": runtime.as_dict(), "metadata": dict(metadata or {})}
    (runtime.manifests / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return runtime


def sync_atomic(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return destination
