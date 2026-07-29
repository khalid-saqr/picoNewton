from pathlib import Path

import nbformat


def test_step3_notebook_contract() -> None:
    path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "scirep_waveform_susceptibility_step3_colab.ipynb"
    )
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "validate_bootstrap_artifacts" in source
    assert "Step3Config" in source
    assert "run_parent_continuity" in source
    assert "allowed_next_step'] == 4" in source
    prohibited = [
        "interaction_kernel(",
        "susceptibility_functional(",
        "critical_anisotropy(",
    ]
    assert not any(token in source for token in prohibited)
