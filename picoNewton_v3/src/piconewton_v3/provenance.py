"""Repository, notebook and environment provenance utilities."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .model import V2_EXPECTED_BLOB_SHA


def git_blob_sha(path: str | Path) -> str:
    file_path = Path(path)
    payload = file_path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def validate_v2_blob(
    path: str | Path,
    expected_sha: str = V2_EXPECTED_BLOB_SHA,
) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    observed = git_blob_sha(file_path)
    if observed != expected_sha:
        raise RuntimeError(
            f"picoNewton_v2.ipynb blob mismatch: expected {expected_sha}, observed {observed}"
        )
    return {
        "path": str(file_path.resolve()),
        "expected_git_blob_sha": expected_sha,
        "observed_git_blob_sha": observed,
        "passed": True,
    }


def strip_notebook_outputs(source: str | Path, destination: str | Path) -> dict[str, Any]:
    try:
        import nbformat
    except ImportError as exc:
        raise RuntimeError("nbformat is required for notebook provenance") from exc
    source_path = Path(source)
    destination_path = Path(destination)
    notebook = nbformat.read(source_path, as_version=4)
    code_cells = 0
    output_items = 0
    for cell in notebook.cells:
        if cell.cell_type == "code":
            code_cells += 1
            output_items += len(cell.get("outputs", []))
            cell["outputs"] = []
            cell["execution_count"] = None
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination_path)
    return {
        "source": str(source_path.resolve()),
        "destination": str(destination_path.resolve()),
        "code_cells": code_cells,
        "removed_output_items": output_items,
        "source_git_blob_sha": git_blob_sha(source_path),
        "stripped_sha256": hashlib.sha256(destination_path.read_bytes()).hexdigest(),
    }


def execute_stripped_v2(
    stripped_notebook: str | Path,
    executed_notebook: str | Path,
    *,
    timeout_s: int = 1800,
    kernel_name: str = "python3",
) -> dict[str, Any]:
    """Cold-execute the output-stripped v2 notebook without changing source cells.

    This is intentionally a publication-profile operation. It may install the
    packages requested by v2 and can therefore be expensive in a clean Colab
    runtime. A failure blocks the publication profile and is written to the run
    provenance rather than silently ignored.
    """
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        raise RuntimeError("nbformat and nbclient are required for cold execution") from exc

    source = Path(stripped_notebook)
    destination = Path(executed_notebook)
    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout_s,
        kernel_name=kernel_name,
        allow_errors=False,
        record_timing=True,
    )
    client.execute(cwd=str(source.parent))
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    return {
        "executed_notebook": str(destination.resolve()),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "passed": True,
    }


def git_commit_or_unknown(repo_root: str | Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(repo_root)), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for module_name in ["numpy", "scipy", "pandas", "matplotlib", "h5py", "nbformat"]:
        try:
            module = __import__(module_name)
            packages[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            packages[module_name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def write_json(path: str | Path, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return output
