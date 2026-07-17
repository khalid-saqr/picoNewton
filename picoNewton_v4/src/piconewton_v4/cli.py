"""Command-line entry point for picoNewton_v4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow_step4 import run_step4
from .workflow_step5 import run_step5
from .workflow_step6 import run_step6


def _package_root(repo_root: Path) -> Path:
    return repo_root / "picoNewton_v4" if (repo_root / "picoNewton_v4").exists() else repo_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-step4", action="store_true")
    parser.add_argument("--run-step5", action="store_true")
    parser.add_argument("--run-step6", action="store_true")
    parser.add_argument("--profile", choices=("quick", "publication"), default="quick")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--hydrodynamic-root",
        type=Path,
        help="Existing Step 4 output directory containing six_artery_hydrodynamics.npz.",
    )
    args = parser.parse_args()

    package_root = _package_root(args.repo_root)
    selected = sum((args.smoke, args.run_step4, args.run_step5, args.run_step6))
    if selected > 1:
        parser.error("select only one action")

    if args.smoke:
        from .sources import validate_cellml_sources

        report = validate_cellml_sources(package_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1
    if args.run_step4:
        output = args.output or package_root / "outputs" / f"step4_{args.profile}"
        manifest = run_step4(package_root=package_root, output_root=output, profile=args.profile)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["status"] == "passed" else 1
    if args.run_step5:
        output = args.output or package_root / "outputs" / "step5_source_model"
        manifest = run_step5(package_root=package_root, output_root=output)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["status"] == "passed" else 1
    if args.run_step6:
        output = args.output or package_root / "outputs" / f"step6_{args.profile}"
        manifest = run_step6(
            package_root=package_root,
            output_root=output,
            profile=args.profile,
            hydrodynamic_root=args.hydrodynamic_root,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["status"] == "passed" else 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
