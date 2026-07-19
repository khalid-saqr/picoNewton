#!/usr/bin/env python3
"""Run picoNewton_v4 from a source checkout."""
from __future__ import annotations

import argparse
from pathlib import Path

from piconewton_v4.reporting import generate_standard_figures
from piconewton_v4.validation import validate_output_directory
from piconewton_v4.workflow import run_workflow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--scan", action="store_true")
    args = parser.parse_args()

    run_workflow(
        package_root=args.package_root,
        output_root=args.output,
        profile=args.profile,
        calibration_path=args.calibration,
        run_scan=args.scan,
    )
    generate_standard_figures(args.output, args.output / "figures")
    report = validate_output_directory(args.output)
    print(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
