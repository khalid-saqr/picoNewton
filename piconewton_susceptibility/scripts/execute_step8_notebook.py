from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    NotebookClient(
        notebook, timeout=args.timeout, kernel_name="python3"
    ).execute(cwd=args.notebook.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
