"""Initialize a source-validated successor workspace without running science calculations."""
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

    bootstrap_dir = "bootstrap/step2"
    store.write_json(f"{bootstrap_dir}/source_validation.json", source_validation)
    manifest = {
        "schema_version": "1.0.0",
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
        "development_skip_parent_validation": config.development_skip_parent_validation,
        "claim_bearing": bool(source_validation["passed"]),
        "scientific_calculations_run": False,
        "scientific_calculations_authorized": False,
        "allowed_next_step": 3 if source_validation["passed"] else None,
    }
    manifest_path = store.write_json(f"{bootstrap_dir}/bootstrap_manifest.json", manifest)
    checksums_path = store.write_checksums(bootstrap_dir, "checksums.sha256")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "source_validation_path": str(output_root / bootstrap_dir / "source_validation.json"),
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
