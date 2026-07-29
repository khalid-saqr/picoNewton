from pathlib import Path

import nbformat


def test_step6_notebook_is_valid_and_scoped() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "scirep_waveform_susceptibility_step6_colab.ipynb"
    )
    notebook = nbformat.read(path, as_version=4)
    text = "\n".join("".join(cell.get("source", "")) for cell in notebook.cells)
    assert "piconewton-susceptibility-step6" in text
    assert "--step5-root" in text and "--step4-root" in text
    assert "critical_anisotropy.csv" in text
    assert "cross vessel and waveform" in text.lower()
