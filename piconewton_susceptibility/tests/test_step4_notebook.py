from pathlib import Path

import nbformat


def test_step4_notebook_is_valid_and_has_scope_boundary() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "scirep_waveform_susceptibility_step4_colab.ipynb"
    )
    notebook = nbformat.read(path, as_version=4)
    text = "\n".join("".join(cell.get("source", "")) for cell in notebook.cells)
    assert "piconewton-susceptibility-step4" in text
    assert "step3_manifest.json" in text
    assert "does not construct the harmonic-interaction kernel" in text
