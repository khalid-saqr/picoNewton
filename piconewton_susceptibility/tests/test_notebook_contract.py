from pathlib import Path

import nbformat


def test_colab_notebook_is_bootstrap_only() -> None:
    path = Path(__file__).resolve().parents[1] / "notebooks" / "scirep_waveform_susceptibility_colab.ipynb"
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join("".join(cell.source) for cell in notebook.cells)
    assert "drive.mount" in source
    assert "successor/scirep-waveform-susceptibility" in source
    assert "\"pip\", \"install\", \"-e\"" in source
    assert "bootstrap_environment" in source
    assert "scientific_calculations_run" in source
    forbidden = [
        "solve_harmonic(",
        "compute_hydrodynamics(",
        "interaction_kernel",
        "critical_anisotropy",
        "Phi_2",
    ]
    for token in forbidden:
        assert token not in source
