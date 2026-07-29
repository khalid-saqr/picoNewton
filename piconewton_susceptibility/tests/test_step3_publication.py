from pathlib import Path

from piconewton_susceptibility.continuity import Step3Config, run_parent_continuity


def _gate(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "source_validation.json").write_text('{"passed": true}')
    (root / "runtime_validation.json").write_text('{"passed": true}')
    (root / "bootstrap_manifest.json").write_text(
        '{"status":"complete","claim_bearing":true,"storage_mode":"local"}'
    )
    (root / "completion_gate.json").write_text('{"passed":true,"allowed_next_step":3}')
    (root / "checksums.sha256").write_text("")
    return root


def test_publication_resolution_closes_step3(tmp_path: Path) -> None:
    result = run_parent_continuity(tmp_path / "out", _gate(tmp_path / "step2"), Step3Config())
    assert result["manifest"]["status"] == "complete"
    assert result["manifest"]["allowed_next_step"] == 4
    assert result["manifest"]["gates"]["passed"] is True
    assert result["historical"]["historical_baseline_passed"].all()
    assert result["convergence"]["max_total_relative_change"].max() <= 0.01
    assert result["convergence"]["max_excess_relative_change"].max() <= 0.01
