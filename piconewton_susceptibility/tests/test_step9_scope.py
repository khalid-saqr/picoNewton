from pathlib import Path


def test_step9_has_no_sensor_or_refit_path():
    root = Path(__file__).resolve().parents[1] / "src" / "piconewton_susceptibility"
    text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in (
            "robustness_core.py",
            "robustness_setup.py",
            "robustness_paths.py",
            "robustness_checks.py",
            "robustness_exact.py",
            "robustness_eta.py",
            "robustness_resolution.py",
            "robustness_claims.py",
            "robustness_metrics.py",
            "robustness_reporting.py",
            "robustness_workflow.py",
        )
    )
    assert "piconewton_v3.sensor" not in text
    assert "fit_scalar_moment" not in text
    assert "frozen_law_not_refitted" in text
    assert "biological_endpoint_model_run" in text
