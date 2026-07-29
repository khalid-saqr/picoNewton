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
    client = NotebookClient(notebook, timeout=1800, kernel_name="python3")
    client.execute(cwd=args.notebook.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
