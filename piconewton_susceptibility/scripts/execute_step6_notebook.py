from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    notebook_path = package_root / "notebooks" / "scirep_waveform_susceptibility_step6_colab.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(package_root.parent)}},
    ).execute()
    output = package_root / "notebooks" / "scirep_waveform_susceptibility_step6_executed.ipynb"
    nbformat.write(notebook, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
