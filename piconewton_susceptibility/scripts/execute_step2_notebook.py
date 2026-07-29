#!/usr/bin/env python3
"""Execute and verify the Step 2 notebook in a clean local kernel."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    repo_root = package_root.parent
    notebook_path = package_root / "notebooks" / "scirep_waveform_susceptibility_colab.ipynb"
    executed_path = package_root / "notebooks" / "_executed_step2_ci.ipynb"
    output_root = repo_root / "piconewton_susceptibility_outputs"

    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(repo_root)}},
    )
    client.execute()
    nbformat.write(notebook, executed_path)

    bootstrap_root = output_root / "bootstrap" / "step2"
    gate = json.loads((bootstrap_root / "completion_gate.json").read_text(encoding="utf-8"))
    if not gate.get("passed") or gate.get("allowed_next_step") != 3:
        raise RuntimeError(f"Step 2 completion gate failed: {gate}")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
