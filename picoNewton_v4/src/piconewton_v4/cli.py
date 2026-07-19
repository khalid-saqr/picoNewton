"""Command-line entry point for picoNewton_v4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import run_workflow


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the six-artery anisotropic Womersley-to-Piezo1 workflow."
    )
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--hydrodynamic-root", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--require-calibrated", action="store_true")
    parser.add_argument(
        "--run-scan",
        action="store_true",
        help="Run the optional localization/channel-count sensitivity scan.",
    )
    args = parser.parse_args()

    root = args.package_root.resolve()
    if not (root / "pyproject.toml").exists() and (root / "picoNewton_v4").exists():
        root = root / "picoNewton_v4"

    manifest = run_workflow(
        package_root=root,
        output_root=args.output,
        run_scan=args.run_scan,
        profile=args.profile,
        hydrodynamic_root=args.hydrodynamic_root,
        calibration_path=args.calibration,
        require_calibrated=args.require_calibrated,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0 if str(manifest["status"]).startswith("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
