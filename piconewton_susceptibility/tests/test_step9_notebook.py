import json
from pathlib import Path


def test_step9_notebook_is_clean_and_complete():
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "scirep_waveform_susceptibility_step9_colab.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code
    assert all(cell.get("execution_count") is None for cell in code)
    assert all(not cell.get("outputs") for cell in code)
    source = "\n".join("".join(cell["source"]) for cell in code)
    assert "piconewton-susceptibility-step9" in source
    assert "step9_manifest.json" in source
    assert "allowed_next_step" in source
