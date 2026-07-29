from pathlib import Path


def test_step6_scope_excludes_later_experiment_layers() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "piconewton_susceptibility"
    text = "\n".join(
        (root / name).read_text(encoding="utf-8").lower()
        for name in ("susceptibility_core.py", "susceptibility_workflow.py")
    )
    forbidden = (
        "low_rank",
        "sobol",
        "mechanosensor",
        "piezo",
        "compliant_wall",
        "nonreciprocal",
        "phase_scrambling",
    )
    assert not any(token in text for token in forbidden)
    assert '"crossed_waveforms_or_reduction_run": false' in text
    assert '"exposure_kernel_or_biological_threshold_used": false' in text
