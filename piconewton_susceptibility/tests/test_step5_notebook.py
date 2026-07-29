import json
from pathlib import Path


def test_step5_notebook_has_gate_and_stop_boundary() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "scirep_waveform_susceptibility_step5_colab.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    assert "piconewton-susceptibility-step5" in source
    assert "step5_manifest.json" in source
    assert "authorizes Step 6" in source
    assert "does not construct the susceptibility functional" in source
