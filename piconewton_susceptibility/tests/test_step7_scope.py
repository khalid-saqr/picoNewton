from pathlib import Path


def test_step7_scope_excludes_later_work():
    root = Path(__file__).parents[1] / "src" / "piconewton_susceptibility"
    text = (root / "experiments_workflow.py").read_text(encoding="utf-8").lower()
    assert '"low_rank_or_constitutive_robustness_run": false' in text
    for prohibited in ("piezo", "mechanosensor", "sobol", "machine learning"):
        assert prohibited not in text
