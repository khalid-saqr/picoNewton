from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiments_core import Step7Config
from .experiments_workflow import run_waveform_experiments


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step 7 crossed susceptibility experiments")
    parser.add_argument("--step6-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("quick", "publication"), default="publication")
    args = parser.parse_args()
    config = (
        Step7Config()
        if args.profile == "publication"
        else Step7Config(
            profile="quick",
            radial_order=48,
            time_points=256,
            quadrature_nodes=48,
        )
    )
    result = run_waveform_experiments(args.output, args.step6_root, config)
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))
    return 0 if result["manifest"]["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
