from pathlib import Path


def test_step5_scope_excludes_later_scientific_layers() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "piconewton_susceptibility"
    text = "\n".join(
        (root / name).read_text(encoding="utf-8").lower()
        for name in ("kernel_core.py", "kernel_workflow.py")
    )
    forbidden = (
        "critical_anisotropy",
        "low_rank",
        "mechanosensor",
        "piezo",
        "sobol",
        "compliant_wall",
    )
    assert not any(token in text for token in forbidden)
    assert '"exposure_used_in_kernel": false' in text
