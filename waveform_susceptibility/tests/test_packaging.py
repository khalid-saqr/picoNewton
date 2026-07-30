from pathlib import Path


def test_parent_runtime_dependency_is_declared():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '"piconewton-v3>=0.1.0"' in pyproject
