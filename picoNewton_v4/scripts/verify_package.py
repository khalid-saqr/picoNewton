#!/usr/bin/env python3
"""Static package and notebook integrity checks."""
from __future__ import annotations

import ast
import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).parents[1]
    required = [
        root / "pyproject.toml",
        root / "notebooks" / "picoNewton_v4_colab.ipynb",
        root / "src" / "piconewton_v4" / "workflow.py",
        root / "src" / "piconewton_v4" / "hydrodynamics.py",
        root / "src" / "piconewton_v4" / "membrane.py",
        root / "src" / "piconewton_v4" / "piezo1.py",
        root / "configs" / "literature_calibration.json",
        root / "data" / "ground_truth_arteries.csv",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    for path in (root / "src").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in (root / "configs").glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    notebook = json.loads((root / "notebooks" / "picoNewton_v4_colab.ipynb").read_text(encoding="utf-8"))
    if len(notebook.get("cells", [])) < 20:
        raise ValueError("notebook documentation is incomplete")
    if any(cell.get("outputs") for cell in notebook["cells"] if cell.get("cell_type") == "code"):
        raise ValueError("notebook must be distributed without stored outputs")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"notebook-cell-{index}")
    print("Package integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
