"""Initialize and independently validate a Step 2 successor workspace."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .parent import load_parent_api
from .provenance import validate_parent_source
from .source_registry import load_source_registry
from .validation import (
    storage_round_trip_probe,
    validate_bootstrap_artifacts,
    verify_checksum_manifest,
)


@dataclass(frozen=True)
class BootstrapConfig:
    repo_root: Path
    storage_mode: str = "auto"
    drive_subdir: str = "MyDrive/picoNewton_susceptibility"
    local_root: Path = Path("./piconewton_susceptibility_outputs")
    development_skip_parent_validation: bool = False


def bootstrap_environment(config: BootstrapConfig) -> dict[str, Any]:
    registry = load_source_registry()
    api = load_parent_api(registry)

    if config.development_skip_parent_validation:
        source_validation = {
            "repository_root": str(config.repo_root.resolve()),
            "registry_sha256": registry.sha256,
            "passed": False,
            "development_skip": True,
            "files": [],
        }
    else:
        source_validation = validate_parent_source(config.repo_root, registry)
        if not source_validation["passed"]:
            failed = [item["path"] for item in source_validation["files"] if not item["passed"]]
            raise RuntimeError(f"parent source validation failed for: {failed}")

    output_root, resolved_mode = api.resolve_study_root(
        mode=config.storage_mode,
        drive_subdir=config.drive_subdir,
        local_root=config.local_root,
    )
    store = api.StudyStore(output_root)
    store.initialize_layout()

    storage_probe = storage_round_trip_probe(output_root)
    if not storage_probe["passed"]:
        raise RuntimeError(f"storage round-trip validation failed: {storage_probe}")

    bootstrap_dir = "bootstrap/step2"
    bootstrap_root = Path(output_root) / bootstrap_dir
    source_path = store.write_json(f"{bootstrap_dir}/source_validation.json", source_validation)

    runtime_validation = {
        "schema_version": "1.0.0",
        "step": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "storage_mode": resolved_mode,
        "storage_root": str(output_root),
        "storage_round_trip": storage_probe,
        "parent_api": {
            "study_store_available": callable(api.StudyStore),
            "storage_resolver_available": callable(api.resolve_study_root),
        },
    }
    runtime_validation["passed"] = bool(
        runtime_validation["storage_round_trip"]["passed"]
        and runtime_validation["parent_api"]["study_store_available"]
        and runtime_validation["parent_api"]["storage_resolver_available"]
    )
    runtime_path = store.write_json(f"{bootstrap_dir}/runtime_validation.json", runtime_validation)

    manifest = {
        "schema_version": "1.1.0",
        "step": 2,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "storage_mode": resolved_mode,
        "storage_root": str(output_root),
        "config": {
            **asdict(config),
            "repo_root": str(config.repo_root),
            "local_root": str(config.local_root),
        },
        "source_registry_sha256": registry.sha256,
        "parent_source_validation_passed": bool(source_validation["passed"]),
        "runtime_validation_passed": bool(runtime_validation["passed"]),
        "development_skip_parent_validation": config.development_skip_parent_validation,
        "claim_bearing": bool(source_validation["passed"] and runtime_validation["passed"]),
        "scientific_calculations_run": False,
        "scientific_calculations_authorized": False,
        "authorization_record": f"{bootstrap_dir}/completion_gate.json",
    }
    manifest_path = store.write_json(f"{bootstrap_dir}/bootstrap_manifest.json", manifest)

    checksums_path = store.write_checksums(bootstrap_dir, "checksums.sha256")
    integrity_precheck = verify_checksum_manifest(
        bootstrap_root,
        required_artifacts={
            "source_validation.json",
            "runtime_validation.json",
            "bootstrap_manifest.json",
        },
    )
    completion_gate = {
        "schema_version": "1.0.0",
        "step": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_validation_passed": bool(source_validation["passed"]),
        "runtime_validation_passed": bool(runtime_validation["passed"]),
        "integrity_precheck_passed": bool(integrity_precheck["passed"]),
        "claim_bearing": bool(manifest["claim_bearing"]),
        "passed": bool(
            source_validation["passed"]
            and runtime_validation["passed"]
            and integrity_precheck["passed"]
        ),
        "allowed_next_step": (
            3
            if source_validation["passed"]
            and runtime_validation["passed"]
            and integrity_precheck["passed"]
            else None
        ),
        "scientific_calculations_authorized_inside_step2": False,
    }
    gate_path = store.write_json(f"{bootstrap_dir}/completion_gate.json", completion_gate)
    checksums_path = store.write_checksums(bootstrap_dir, "checksums.sha256")

    final_validation = validate_bootstrap_artifacts(
        bootstrap_root,
        require_claim_bearing=not config.development_skip_parent_validation,
        expected_storage_mode=resolved_mode,
    )
    if not final_validation["passed"]:
        raise RuntimeError(f"final Step 2 artifact validation failed: {final_validation['checks']}")

    return {
        "manifest": manifest,
        "completion_gate": completion_gate,
        "validation": final_validation,
        "output_root": str(output_root),
        "bootstrap_root": str(bootstrap_root),
        "manifest_path": str(manifest_path),
        "source_validation_path": str(source_path),
        "runtime_validation_path": str(runtime_path),
        "completion_gate_path": str(gate_path),
        "checksums_path": str(checksums_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--storage", choices=["auto", "drive", "local"], default="auto")
    parser.add_argument("--drive-subdir", default="MyDrive/picoNewton_susceptibility")
    parser.add_argument("--local-root", type=Path, default=Path("./piconewton_susceptibility_outputs"))
    parser.add_argument(
        "--development-skip-parent-validation",
        action="store_true",
        help="Non-claim-bearing isolated smoke-test mode; never use for scientific runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = bootstrap_environment(
        BootstrapConfig(
            repo_root=args.repo_root,
            storage_mode=args.storage,
            drive_subdir=args.drive_subdir,
            local_root=args.local_root,
            development_skip_parent_validation=args.development_skip_parent_validation,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
