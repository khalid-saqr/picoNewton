from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable


@dataclass(frozen=True)
class Step10Config:
    profile: str = "publication"
    figure_dpi: int = 300
    figure_formats: tuple[str, ...] = ("png", "pdf")
    include_step_directories: tuple[str, ...] = (
        "bootstrap/step2",
        "step3_parent_continuity",
        "step4_perturbation",
        "step5_harmonic_kernel",
        "step6_susceptibility",
        "step7_waveform_experiments",
        "step8_reduced_law",
        "step9_robustness_claim_lock",
    )

    def validate(self) -> None:
        if self.profile not in {"quick", "publication"}:
            raise ValueError("profile must be quick or publication")
        if self.figure_dpi < 150:
            raise ValueError("figure_dpi must be at least 150")
        allowed = {"png", "pdf", "svg"}
        if not self.figure_formats or not set(self.figure_formats).issubset(allowed):
            raise ValueError("figure_formats must be selected from png, pdf and svg")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def validate_manifest_root(
    root: str | Path,
    step: int,
    expected_next: int | None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    gate_path = root / f"step{step}_gate.json"
    manifest_path = root / f"step{step}_manifest.json"
    if not gate_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"Step 10 requires Step {step} gate and manifest at {root}"
        )
    gate = load_json(gate_path)
    manifest = load_json(manifest_path)
    if not gate.get("passed"):
        raise RuntimeError(f"Step {step} gate is not passing")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Step {step} manifest is not complete")
    if expected_next is not None:
        if manifest.get("allowed_next_step") != expected_next:
            raise RuntimeError(
                f"Step {step} manifest does not authorise Step {expected_next}"
            )
    for name, record in manifest.get("files", {}).items():
        candidate = root / name
        if not candidate.is_file():
            raise RuntimeError(f"Step {step} artifact is missing: {name}")
        if sha256(candidate) != record.get("sha256"):
            raise RuntimeError(f"Step {step} checksum failed: {name}")
    return {"root": root, "gate": gate, "manifest": manifest}


def validate_step2(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    required = {
        "source_validation.json",
        "runtime_validation.json",
        "bootstrap_manifest.json",
        "completion_gate.json",
        "checksums.sha256",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise RuntimeError(f"Step 2 artifacts are missing: {missing}")

    source = load_json(root / "source_validation.json")
    runtime = load_json(root / "runtime_validation.json")
    manifest = load_json(root / "bootstrap_manifest.json")
    gate = load_json(root / "completion_gate.json")
    if not source.get("passed") or not runtime.get("passed"):
        raise RuntimeError("Step 2 source or runtime validation is not passing")
    if manifest.get("status") != "complete" or not manifest.get("claim_bearing"):
        raise RuntimeError("Step 2 manifest is not claim-bearing and complete")
    if not gate.get("passed") or gate.get("allowed_next_step") != 3:
        raise RuntimeError("Step 2 completion gate does not authorise Step 3")

    parsed: dict[str, str] = {}
    checksum_text = (root / "checksums.sha256").read_text(encoding="utf-8")
    for line in checksum_text.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(None, 1)
        parsed[name.strip().lstrip("*")] = digest
    checksum_targets = required - {"checksums.sha256", "completion_gate.json"}
    for name in checksum_targets:
        if parsed.get(name) != sha256(root / name):
            raise RuntimeError(f"Step 2 checksum failed: {name}")
    return {
        "root": root,
        "gate": gate,
        "manifest": manifest,
        "source": source,
        "runtime": runtime,
    }


def validate_step9(root: str | Path) -> dict[str, Any]:
    result = validate_manifest_root(root, 9, 10)
    claim_path = result["root"] / "claim_lock.json"
    if not claim_path.is_file():
        raise RuntimeError("Step 10 requires claim_lock.json")
    claim = load_json(claim_path)
    if claim.get("status") != "locked":
        raise RuntimeError("Step 9 claim lock is not locked")
    if claim.get("allowed_next_step") != 10:
        raise RuntimeError("Step 9 claim lock does not authorise Step 10")
    result["claim_lock"] = claim
    return result


def file_records(
    root: str | Path,
    paths: Iterable[str | Path],
) -> dict[str, dict[str, Any]]:
    root = Path(root).resolve()
    records: dict[str, dict[str, Any]] = {}
    for item in paths:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        relative = path.relative_to(root).as_posix()
        records[relative] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return records


def environment_manifest() -> dict[str, Any]:
    packages = {}
    package_names = (
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "h5py",
        "nbformat",
        "nbclient",
    )
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment_keys = (
        "GITHUB_SHA",
        "GITHUB_REF_NAME",
        "COLAB_RELEASE_TAG",
    )
    environment = {
        key: os.environ.get(key)
        for key in environment_keys
        if os.environ.get(key)
    }
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "environment": environment,
    }


def git_provenance(repo_root: str | Path | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"available": False}
    if repo_root is None:
        return payload
    root = Path(repo_root).resolve()
    commands = {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--porcelain"],
        "remote": ["git", "remote", "get-url", "origin"],
    }
    values: dict[str, str | None] = {}
    for name, command in commands.items():
        try:
            values[name] = subprocess.check_output(
                command,
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            values[name] = None
    payload.update(values)
    payload["available"] = bool(values.get("commit"))
    payload["working_tree_clean"] = values.get("status") == ""
    return payload
