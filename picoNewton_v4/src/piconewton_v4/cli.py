"""Command-line entry point for the Step 3 scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sources import validate_cellml_sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        package_root = args.repo_root / "picoNewton_v4" if (args.repo_root / "picoNewton_v4").exists() else args.repo_root
        report = validate_cellml_sources(package_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
