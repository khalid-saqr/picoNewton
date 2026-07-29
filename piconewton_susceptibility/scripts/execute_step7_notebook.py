from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    executed = NotebookClient(notebook, timeout=None, kernel_name="python3").execute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
