from pathlib import Path


def test_step3_source_excludes_later_science() -> None:
    root = Path(__file__).parents[1] / "src" / "piconewton_susceptibility"
    text = (root / "continuity.py").read_text(encoding="utf-8").lower()
    prohibited = [
        "interaction_kernel",
        "susceptibility_functional",
        "critical_anisotropy",
        "piezo",
        "sensorconfig",
        "calcium",
    ]
    assert not any(token in text for token in prohibited)
