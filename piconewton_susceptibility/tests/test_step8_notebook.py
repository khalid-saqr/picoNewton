import json
from pathlib import Path


def test_step8_notebook_is_clean_and_ordered():
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "scirep_waveform_susceptibility_step8_colab.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(
        cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "piconewton-susceptibility-step8" in source
    assert "step7_waveform_experiments" in source
    assert "step8_reduced_law" in source
