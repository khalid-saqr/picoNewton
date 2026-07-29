from __future__ import annotations

import argparse
import json
from pathlib import Path

from piconewton_susceptibility.continuity import Step3Config, run_parent_continuity


def _config(profile: str) -> Step3Config:
    if profile == "quick":
        return Step3Config(
            profile="quick",
            radial_order=80,
            time_points=512,
            quadrature_nodes=96,
            radial_checks=(60, 100),
            time_checks=(256, 1024),
            quadrature_checks=(48, 192),
        )
    return Step3Config()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=["quick", "publication"], default="publication"
    )
    parser.add_argument("--development-skip-step2", action="store_true")
    args = parser.parse_args()
    result = run_parent_continuity(
        args.output,
        args.step2_root,
        _config(args.profile),
        require_step2=not args.development_skip_step2,
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))
    return 0 if result["manifest"]["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
