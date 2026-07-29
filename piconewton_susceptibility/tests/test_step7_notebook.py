from pathlib import Path

import nbformat


def test_step7_notebook_is_clean_and_scoped():
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "scirep_waveform_susceptibility_step7_colab.ipynb"
    )
    notebook = nbformat.read(path, as_version=4)
    assert all(
        cell.get("execution_count") is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
    text = "\n".join("".join(cell.source) for cell in notebook.cells)
    assert "piconewton-susceptibility-step7" in text
    assert "step7_manifest.json" in text
    assert "low-rank" not in text.lower()
