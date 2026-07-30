from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[1] / "src" / "piconewton_susceptibility"


def test_step10_contains_no_new_scientific_solver_or_fit():
    names = [
        "publication_core.py",
        "publication_figures.py",
        "publication_tables.py",
        "publication_workflow.py",
    ]
    source = "\n".join((SOURCE_ROOT / name).read_text(encoding="utf-8") for name in names)
    prohibited = [
        "WomersleySolver",
        "solve_harmonic",
        "derive_hierarchy",
        "fit_power_law",
        "fit_scalar_moment",
        "minimize_scalar",
        "brentq",
        "mechanosensor",
        "Piezo",
    ]
    for token in prohibited:
        assert token not in source
    assert '"scientific_claim_modified": False' in source
    assert '"new_scientific_fit_run": False' in source
