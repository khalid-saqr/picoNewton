from pathlib import Path

from piconewton_v4.workflow_step5 import (
    figure3_voltage_clamp,
    numerical_validation,
    run_step5,
)


def test_source_voltage_behavior():
    result = figure3_voltage_clamp()
    assert result["validation"]["positive_voltage_reduces_inactivation"]
    assert result["validation"]["all_probabilities_bounded"]


def test_step5_numerical_validation():
    report = numerical_validation()
    assert report["passed"], report


def test_step5_workflow(tmp_path: Path):
    package_root = Path(__file__).resolve().parents[1]
    report = run_step5(package_root=package_root, output_root=tmp_path / "step5")
    assert report["status"] == "passed", report
    assert report["coupling_executed"] is False
    assert (tmp_path / "step5" / "step5_validation.json").exists()
    assert (tmp_path / "step5" / "figure3_voltage_clamp_summary.csv").exists()
