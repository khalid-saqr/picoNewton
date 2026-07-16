from __future__ import annotations

from pathlib import Path

import nbformat


def test_notebook_is_clean_compilable_and_cited() -> None:
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "notebooks" / "picoNewton_v3_mechanosensory.ipynb"
    notebook = nbformat.read(path, as_version=4)
    assert len(notebook.cells) >= 25
    assert any("parameters" in cell.metadata.get("tags", []) for cell in notebook.cells)
    all_markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    assert "10.1038/s41598-026-47474-x" in all_markdown
    assert "10.1073/pnas.0307804101" in all_markdown
    assert "10.1242/jcs.264456" in all_markdown
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
            compile(cell.source, f"notebook-cell-{index}", "exec")
