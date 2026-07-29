from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from piconewton_susceptibility.perturbation import (
    Step4Config,
    derive_hierarchy,
    run_perturbative_hierarchy,
)
from piconewton_v3 import V2_ARTERY_CASES


def _step3_gate(root: Path) -> Path:
    root.mkdir(parents=True)
    evidence = root / "six_artery_continuity.csv"
    evidence.write_text("artery_id\naortic_root\n", encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    (root / "step3_gate.json").write_text(
        json.dumps({"passed": True, "step": 3}), encoding="utf-8"
    )
    (root / "step3_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "allowed_next_step": 4,
                "files": {
                    evidence.name: {"sha256": digest, "bytes": evidence.stat().st_size}
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def _quick(profile: str = "quick") -> Step4Config:
    return Step4Config(
        profile=profile,
        radial_order=50,
        time_points=256,
        quadrature_nodes=48,
        epsilon_values=(0.01, 0.02, 0.04, 0.08),
        slope_epsilon_max=0.04,
        parity_epsilon=0.04,
        minimum_valid_epsilon=0.04,
    )


def test_rejects_missing_step3_gate(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        run_perturbative_hierarchy(tmp_path / "out", tmp_path / "missing", _quick())


def test_hierarchy_satisfies_boundary_conditions_and_residual() -> None:
    hierarchy = derive_hierarchy(V2_ARTERY_CASES[0], _quick())
    assert hierarchy.max_residual < 1e-10
    assert np.max(np.abs(hierarchy.ut1[[0, -1], :])) < 1e-10
    assert np.max(np.abs(hierarchy.uz2[-1, :])) < 1e-10


def test_quick_run_is_diagnostic_but_covers_six_arteries(tmp_path: Path) -> None:
    result = run_perturbative_hierarchy(
        tmp_path / "out", _step3_gate(tmp_path / "step3"), _quick()
    )
    assert len(result["validity"]) == 6
    assert result["manifest"]["status"] == "failed"
    assert result["manifest"]["allowed_next_step"] is None
    assert result["manifest"]["gates"]["publication_profile"] is False
    assert result["manifest"]["gates"]["parity_passed"] is True
    assert result["manifest"]["gates"]["force_order_passed"] is True


def test_archive_contains_coefficients_and_predictions(tmp_path: Path) -> None:
    result = run_perturbative_hierarchy(
        tmp_path / "out", _step3_gate(tmp_path / "step3"), _quick()
    )
    archive = np.load(tmp_path / "out" / "perturbation_waveforms.npz")
    for artery_id in result["validity"]["artery_id"]:
        assert f"{artery_id}__force0_n" in archive
        assert f"{artery_id}__force2_n" in archive
        assert f"{artery_id}__uz0" in archive
        assert f"{artery_id}__ut1" in archive
        assert f"{artery_id}__uz2" in archive
