from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reduction_core import Step8Config
from .reduction_workflow import run_reduction_study


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step 8 reduced-law validation")
    parser.add_argument("--step7-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("quick", "publication"), default="publication")
    args = parser.parse_args()
    config = Step8Config(profile=args.profile)
    result = run_reduction_study(args.output, args.step7_root, config)
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))
    return 0 if result["manifest"]["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
