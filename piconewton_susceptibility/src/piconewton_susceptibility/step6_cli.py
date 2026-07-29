from __future__ import annotations

import argparse
import json
from pathlib import Path

from .susceptibility_core import Step6Config
from .susceptibility_workflow import run_susceptibility_inversion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step 6 susceptibility and inversion")
    parser.add_argument("--step5-root", required=True)
    parser.add_argument("--step4-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("quick", "publication"), default="publication")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.profile == "quick":
        config = Step6Config(
            profile="quick",
            radial_order=60,
            time_points=512,
            quadrature_nodes=96,
            validation_epsilons=(0.04, 0.08),
            inversion_verification_epsilons=(0.04,),
        )
    else:
        config = Step6Config()
    result = run_susceptibility_inversion(
        Path(args.output),
        Path(args.step5_root),
        Path(args.step4_root),
        config,
        require_prior_steps=args.profile == "publication",
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))
    return 0 if result["manifest"]["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
