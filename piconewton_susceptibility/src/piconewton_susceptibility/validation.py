"""Independent integrity and storage validation for the Step 2 bootstrap."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path, PurePosixPath
from typing import Any

_REQUIRED_ARTIFACTS = {
    "source_validation.json",
    "runtime_validation.json",
    "bootstrap_manifest.json",
    "completion_gate.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def storage_round_trip_probe(output_root: str | Path) -> dict[str, Any]:
    """Test create, write, rename, read, and delete on the selected storage backend."""
    root = Path(output_root).resolve()
    probe_dir = root / "bootstrap" / "step2"
    probe_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(12)
    temporary = probe_dir / f".storage_probe_{token}.tmp"
    final = probe_dir / f".storage_probe_{token}.json"
    payload = json.dumps({"token": token, "purpose": "step2-storage-round-trip"}, sort_keys=True)
    record: dict[str, Any] = {
        "root": str(root),
        "temporary_path": str(temporary),
        "final_path": str(final),
        "passed": False,
    }
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, final)
        observed = final.read_text(encoding="utf-8")
        record["bytes_written"] = len(payload.encode("utf-8"))
        record["content_match"] = observed == payload
        final.unlink()
        record["delete_confirmed"] = not final.exists()
        record["passed"] = bool(record["content_match"] and record["delete_confirmed"])
    except Exception as exc:  # pragma: no cover - backend-specific failure path
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for path in (temporary, final):
            if path.exists():
                path.unlink()
    return record


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def verify_checksum_manifest(
    bootstrap_root: str | Path,
    *,
    required_artifacts: set[str] | None = None,
) -> dict[str, Any]:
    """Verify every entry in a Step 2 checksum file and reject unsafe paths."""
    root = Path(bootstrap_root).resolve()
    checksum_path = root / "checksums.sha256"
    records: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    listed_paths: set[str] = set()

    if not checksum_path.is_file():
        return {
            "bootstrap_root": str(root),
            "checksum_path": str(checksum_path),
            "records": records,
            "parse_errors": ["checksums.sha256 is missing"],
            "required_artifacts_present": False,
            "passed": False,
        }

    checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(checksum_lines, 1):
        if not raw_line.strip():
            continue
        try:
            expected, relative = raw_line.split("  ", 1)
        except ValueError:
            parse_errors.append(f"line {line_number}: expected '<sha256>  <path>'")
            continue
        candidate = PurePosixPath(relative)
        safe = not candidate.is_absolute() and ".." not in candidate.parts
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            parse_errors.append(f"line {line_number}: invalid SHA-256")
            continue
        if not safe or relative == "checksums.sha256":
            parse_errors.append(f"line {line_number}: unsafe or recursive path '{relative}'")
            continue
        if relative in listed_paths:
            parse_errors.append(f"line {line_number}: duplicate path '{relative}'")
            continue
        listed_paths.add(relative)
        path = root.joinpath(*candidate.parts)
        observed = sha256_file(path) if path.is_file() else None
        records.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "passed": observed == expected,
            }
        )

    required = _REQUIRED_ARTIFACTS if required_artifacts is None else required_artifacts
    required_present = required.issubset(listed_paths)
    passed = not parse_errors and required_present and bool(records) and all(
        item["passed"] for item in records
    )
    return {
        "bootstrap_root": str(root),
        "checksum_path": str(checksum_path),
        "records": records,
        "parse_errors": parse_errors,
        "required_artifacts_present": required_present,
        "passed": passed,
    }


def validate_bootstrap_artifacts(
    bootstrap_root: str | Path,
    *,
    require_claim_bearing: bool = True,
    expected_storage_mode: str | None = None,
) -> dict[str, Any]:
    """Independently reopen and validate the final Step 2 artifact set."""
    root = Path(bootstrap_root).resolve()
    paths = {name: root / name for name in _REQUIRED_ARTIFACTS}
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        return {
            "bootstrap_root": str(root),
            "missing": missing,
            "passed": False,
        }

    source = _read_json(paths["source_validation.json"])
    runtime = _read_json(paths["runtime_validation.json"])
    manifest = _read_json(paths["bootstrap_manifest.json"])
    gate = _read_json(paths["completion_gate.json"])
    checksums = verify_checksum_manifest(root)

    checks = {
        "step_is_2": manifest.get("step") == 2,
        "status_is_complete": manifest.get("status") == "complete",
        "no_science_run": manifest.get("scientific_calculations_run") is False,
        "no_science_authorized_inside_step2": (
            manifest.get("scientific_calculations_authorized") is False
        ),
        "storage_probe_passed": runtime.get("storage_round_trip", {}).get("passed") is True,
        "runtime_validation_passed": runtime.get("passed") is True,
        "source_state_consistent": manifest.get("parent_source_validation_passed")
        is bool(source.get("passed")),
        "gate_integrity_precheck_passed": gate.get("integrity_precheck_passed") is True,
        "final_checksums_passed": checksums.get("passed") is True,
    }
    if expected_storage_mode is not None:
        checks["storage_mode_matches"] = manifest.get("storage_mode") == expected_storage_mode
    if require_claim_bearing:
        checks["claim_bearing"] = manifest.get("claim_bearing") is True
        checks["parent_source_validation_passed"] = source.get("passed") is True
        checks["step3_gate_passed"] = gate.get("passed") is True
        checks["allowed_next_step_is_3"] = gate.get("allowed_next_step") == 3
    else:
        checks["development_state_is_safe"] = (
            manifest.get("claim_bearing") is False
            and gate.get("allowed_next_step") is None
            and gate.get("passed") is False
        )

    return {
        "bootstrap_root": str(root),
        "manifest": manifest,
        "source_validation": source,
        "runtime_validation": runtime,
        "completion_gate": gate,
        "checksum_validation": checksums,
        "checks": checks,
        "passed": all(checks.values()),
    }
