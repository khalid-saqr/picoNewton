"""Provenance and checksum helpers."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import platform
import subprocess
import sys
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_manifest(paths: Iterable[Path], root: Path) -> list[dict[str, object]]:
    return [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(paths)]


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record(repo_root: Path) -> dict[str, object]:
    return {"python": sys.version, "platform": platform.platform(), "git_commit": git_commit(repo_root)}


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
