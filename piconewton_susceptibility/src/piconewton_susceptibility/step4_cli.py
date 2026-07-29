from __future__ import annotations

import argparse
import json
from pathlib import Path

from .perturbation import Step4Config, run_perturbative_hierarchy


def _config(profile: str) -> Step4Config:
    if profile == "quick":
        return Step4Config(
            profile="quick",
            radial_order=60,
            time_points=512,
            quadrature_nodes=96,
        )
    return Step4Config()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step 4 weak-anisotropy hierarchy")
    parser.add_argument("--step3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=["quick", "publication"], default="publication"
    )
    parser.add_argument(
        "--development-skip-step3",
        action="store_true",
        help="Software diagnostics only; cannot produce a passing Step 4 gate.",
    )
    args = parser.parse_args()
    result = run_perturbative_hierarchy(
        args.output,
        args.step3_root,
        _config(args.profile),
        require_step3=not args.development_skip_step3,
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))
    return 0 if result["manifest"]["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
