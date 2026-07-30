# ruff: noqa: E501
from __future__ import annotations

import argparse
import json

from .publication_core import Step10Config
from .publication_workflow import run_publication_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the final Scientific Reports publication archive")
    parser.add_argument("--workflow-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root")
    parser.add_argument("--profile", choices=("quick", "publication"), default="publication")
    args = parser.parse_args()
    config = Step10Config(profile=args.profile, figure_dpi=180 if args.profile == "quick" else 300)
    result = run_publication_archive(args.output, args.workflow_root, args.repo_root, config)
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))
    if not result["manifest"]["workflow_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
