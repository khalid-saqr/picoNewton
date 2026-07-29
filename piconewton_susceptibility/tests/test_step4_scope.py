from pathlib import Path

import piconewton_susceptibility.perturbation as perturbation


def test_step4_source_excludes_later_scientific_implementations() -> None:
    source = Path(perturbation.__file__).read_text(encoding="utf-8").lower()
    prohibited_patterns = [
        "def interaction_kernel",
        "from .interaction_kernel",
        "def critical_anisotropy",
        "from .critical_anisotropy",
        "mechanosensor",
        "piezo",
        "sobol",
        "machine learning",
    ]
    assert not any(pattern in source for pattern in prohibited_patterns)
