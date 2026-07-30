from __future__ import annotations

import argparse
import json

from .robustness_core import Step9Config
from .robustness_workflow import run_robustness_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step 9 robustness and claim lock")
    parser.add_argument("--step8-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("quick", "publication"), default="publication")
    arguments = parser.parse_args()
    config = Step9Config()
    if arguments.profile == "quick":
        config = Step9Config(
            profile="quick",
            radial_order=60,
            quadrature_nodes=96,
            resolution_pairs=((50, 72), (70, 112)),
        )
    result = run_robustness_study(arguments.output, arguments.step8_root, config)
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))
    if result["manifest"]["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
