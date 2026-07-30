import re
from pathlib import Path

import nbformat


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "scirep_waveform_susceptibility_step10_colab.ipynb"
)


def test_step10_notebook_is_clean_and_complete():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
    assert all(
        cell.get("execution_count") is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "drive.mount('/content/drive'" in source
    assert "piconewton-susceptibility-bootstrap" in source
    for step in range(3, 11):
        assert f"piconewton-susceptibility-step{step}" in source
    match = re.search(
        r"PINNED_COMMIT = '([0-9a-f]{40}|__STEP10_IMPLEMENTATION_COMMIT__)'",
        source,
    )
    assert match is not None
    assert "publication_archive.sha256" in source
    assert "workflow_complete" in source
