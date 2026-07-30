from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    client = NotebookClient(notebook, timeout=7200, kernel_name="python3")
    executed = client.execute()
    nbformat.write(executed, args.output)


if __name__ == "__main__":
    main()
