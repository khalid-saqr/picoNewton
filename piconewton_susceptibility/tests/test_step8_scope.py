from pathlib import Path


def test_step8_scope_excludes_forbidden_extensions():
    root = Path(__file__).resolve().parents[1] / "src" / "piconewton_susceptibility"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("reduction*.py"))
    forbidden = ("piezo", "mechanosensor", "sobol", "machine learning", "compliance", "fkdv")
    assert not any(term in text.lower() for term in forbidden)
