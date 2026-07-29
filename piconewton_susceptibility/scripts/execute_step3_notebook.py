from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "scirep_waveform_susceptibility_step3_colab.ipynb"
)
EXECUTED_PATH = PROJECT_ROOT / "notebooks" / "_executed_step3_ci.ipynb"

os.chdir(REPO_ROOT)
notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
client = NotebookClient(notebook, timeout=1800, kernel_name="python3", allow_errors=False)
client.execute(cwd=str(REPO_ROOT))
nbformat.write(notebook, EXECUTED_PATH)
