from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from piconewton_susceptibility.continuity import Step3Config, run_parent_continuity


def make_gate(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "source_validation.json").write_text('{"passed": true}')
    (root / "runtime_validation.json").write_text('{"passed": true}')
    (root / "bootstrap_manifest.json").write_text(
        '{"status":"complete","claim_bearing":true,"storage_mode":"local"}'
    )
    (root / "completion_gate.json").write_text('{"passed":true,"allowed_next_step":3}')
    (root / "checksums.sha256").write_text("")
    return root


def quick_config() -> Step3Config:
    return Step3Config(
        profile="quick",
        radial_order=50,
        time_points=256,
        quadrature_nodes=48,
        radial_checks=(40, 60),
        time_checks=(128, 512),
        quadrature_checks=(24, 96),
    )


def test_rejects_missing_step2_gate(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        run_parent_continuity(tmp_path / "out", tmp_path / "missing", quick_config())


def test_quick_run_is_diagnostic_and_complete_for_all_arteries(tmp_path: Path) -> None:
    result = run_parent_continuity(
        tmp_path / "out", make_gate(tmp_path / "step2"), quick_config()
    )
    assert len(result["summary"]) == 6
    assert result["manifest"]["status"] == "failed"
    assert result["manifest"]["allowed_next_step"] is None
    assert result["manifest"]["gates"]["publication_profile"] is False
    assert result["summary"]["anisotropic_exposure_nonnegative"].all()
    assert result["summary"]["isotropic_exposure_nonnegative"].all()
    assert set(result["historical"]["historical_role"]) == {"lineage_only"}


def test_publication_profile_passes_all_step3_gates(tmp_path: Path) -> None:
    config = Step3Config(
        radial_order=80,
        time_points=512,
        quadrature_nodes=96,
        radial_checks=(60, 100),
        time_checks=(256, 1024),
        quadrature_checks=(48, 192),
    )
    result = run_parent_continuity(tmp_path / "out", make_gate(tmp_path / "step2"), config)
    assert result["manifest"]["allowed_next_step"] is None
    assert not result["manifest"]["gates"]["historical_baseline_passed"]
    assert result["manifest"]["gates"]["six_arteries_complete"]
    assert result["manifest"]["gates"]["mechanics_closure_passed"]
    assert result["convergence"]["max_excess_relative_change"].max() < 0.01


def test_waveform_archive_contains_signed_and_exposure_controls(tmp_path: Path) -> None:
    result = run_parent_continuity(
        tmp_path / "out", make_gate(tmp_path / "step2"), quick_config()
    )
    archive = np.load(tmp_path / "out" / "six_artery_waveforms.npz")
    for artery_id in result["summary"]["artery_id"]:
        assert f"{artery_id}__signed_anisotropic_n" in archive
        assert f"{artery_id}__exposure_anisotropic_n" in archive
        assert f"{artery_id}__signed_isotropic_n" in archive
        assert f"{artery_id}__exposure_isotropic_n" in archive
        assert f"{artery_id}__signed_excess_n" in archive
        assert f"{artery_id}__exposure_excess_n" in archive


def test_manifest_hashes_are_recorded(tmp_path: Path) -> None:
    run_parent_continuity(tmp_path / "out", make_gate(tmp_path / "step2"), quick_config())
    manifest = json.loads((tmp_path / "out" / "step3_manifest.json").read_text())
    assert manifest["files"]
    assert all(len(record["sha256"]) == 64 for record in manifest["files"].values())
