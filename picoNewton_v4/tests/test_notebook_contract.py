from pathlib import Path
import json
import re


def _cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else str(source)


def test_colab_notebook_runtime_contract():
    root = Path(__file__).parents[1]
    notebook = json.loads(
        (root / "notebooks" / "picoNewton_v4_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    assert len(notebook["cells"]) >= 20

    markdown = "\n".join(
        _cell_source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    code = "\n".join(_cell_source(cell) for cell in code_cells)

    assert "drive.mount('/content/drive')" in code
    assert "picoNewton_v4_runtime" in code
    assert "datetime.now(timezone.utc)" in code

    # Check the commands semantically rather than depending on quote style or
    # whitespace produced by notebook serialization or manual formatting.
    normalized_code = re.sub(r"\s+", "", code.replace('"', "'"))
    assert "'git','clone'" in normalized_code
    assert "'pip','install'" in normalized_code
    assert "run_workflow" in code

    assert "Scientific question" in markdown
    assert "Vector-resolved membrane" in markdown

    # The historical notebook is intentionally committed with its successful
    # Colab outputs. Saved outputs are allowed; saved Python error outputs are not.
    saved_errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not saved_errors
