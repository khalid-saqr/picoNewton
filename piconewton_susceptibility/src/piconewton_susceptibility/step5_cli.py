from __future__ import annotations

import argparse
import json
from pathlib import Path

from .kernel import Step5Config, run_harmonic_kernel


def _config(profile: str) -> Step5Config:
    if profile == "quick":
        return Step5Config(
            profile="quick",
            radial_order=50,
            time_points=512,
            quadrature_nodes=96,
            closure_tolerance=1e-9,
        )
    return Step5Config()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step 5 harmonic-interaction kernel")
    parser.add_argument("--step4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["quick", "publication"], default="publication")
    parser.add_argument(
        "--development-skip-step4",
        action="store_true",
        help="Software diagnostics only; cannot produce a passing Step 5 gate.",
    )
    args = parser.parse_args()
    result = run_harmonic_kernel(
        args.output,
        args.step4_root,
        _config(args.profile),
        require_step4=not args.development_skip_step4,
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))
    return 0 if result["manifest"]["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
