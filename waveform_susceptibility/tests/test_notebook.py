import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "waveform_susceptibility_colab.ipynb"
)


def test_colab_notebook_is_clean_and_reproducible():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert all(cell.get("execution_count") is None for cell in notebook["cells"])
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "drive.mount('/content/drive')" in source
    assert "uuid4().hex[:8]" in source
    assert "PICONEWTON_REF" in source
    assert "--figure-dpi" in source and "'600'" in source
    assert "runtime_metadata.json" in source
    assert "checksums.sha256" in source
    assert "waveform_susceptibility_results.zip" in source
    assert "figure_manifest.json" in source
