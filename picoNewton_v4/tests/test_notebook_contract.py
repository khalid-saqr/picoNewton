from pathlib import Path
import json


def test_colab_notebook_runtime_contract():
    root = Path(__file__).parents[1]
    notebook = json.loads(
        (root / "notebooks" / "picoNewton_v4_colab.ipynb").read_text(encoding="utf-8")
    )
    assert len(notebook["cells"]) >= 20
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "drive.mount('/content/drive')" in code
    assert "picoNewton_v4_runtime" in code
    assert "datetime.now(timezone.utc)" in code
    assert "git', 'clone'" in code
    assert "pip', 'install'" in code
    assert "run_workflow" in code
    assert "Scientific question" in markdown
    assert "Vector-resolved membrane" in markdown
    assert all(
        not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
