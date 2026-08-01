from pathlib import Path


def test_parent_runtime_dependency_is_declared():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert '"piconewton-v3>=0.1.0"' in pyproject
    assert 'version = "1.0.1"' in pyproject


def test_readme_identifies_the_manuscript_ground_truth():
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    assert (
        "Harmonic interactions shape anisotropy-induced transverse force in arterial blood flow"
    ) in readme
    assert "Figures 1--5 and Supplementary Figure S1" in readme
