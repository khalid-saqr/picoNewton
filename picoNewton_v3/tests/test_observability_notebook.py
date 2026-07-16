from __future__ import annotations

from pathlib import Path

import nbformat


def test_observability_notebook_is_clean_compilable_and_scientifically_structured() -> None:
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "notebooks" / "picoNewton_v3_mechanosensory_observability.ipynb"
    notebook = nbformat.read(path, as_version=4)

    assert len(notebook.cells) >= 25
    assert any("parameters" in cell.metadata.get("tags", []) for cell in notebook.cells)

    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    required_text = {
        "10.1038/s41598-026-47474-x",
        "Hydrodynamic nonredundancy before the sensor",
        "Analytical and nonlinear mechanosensory transfer",
        "Leave-one-artery-out WSS-surrogate competition",
        "Robustness, gate margins, and retained claims",
    }
    assert all(text in markdown for text in required_text)

    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for section in range(1, 9):
        assert f"section_{section:02d}_" in code

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
            compile(cell.source, f"observability-notebook-cell-{index}", "exec")

    section_root = project_root / "src" / "piconewton_v3" / "observability_notebook"
    section_files = sorted(section_root.glob("section_*.py"))
    assert len(section_files) == 8
    for section_file in section_files:
        compile(section_file.read_text(encoding="utf-8"), str(section_file), "exec")
