from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from piconewton_susceptibility.kernel import Step5Config, run_harmonic_kernel
from piconewton_susceptibility.kernel_core import (
    direct_second_order_waveform,
    near_wall_basis,
    unit_perturbation_response,
)
from piconewton_v3 import V2_ARTERY_CASES


def _step4_evidence(root: Path, config: Step5Config) -> Path:
    root.mkdir(parents=True)
    arrays = {}
    for case in V2_ARTERY_CASES:
        basis = unit_perturbation_response(case, config)
        _, basis = near_wall_basis(case, basis, config)
        arrays[f"{case.artery_id}__force2_n"] = direct_second_order_waveform(
            case, basis, case.harmonic_coefficients, config.time_points
        )
    archive = root / "perturbation_waveforms.npz"
    np.savez_compressed(archive, **arrays)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (root / "step4_gate.json").write_text('{"passed":true}', encoding="utf-8")
    (root / "step4_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "allowed_next_step": 5,
                "files": {archive.name: {"sha256": digest}},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_reduced_publication_profile_closes_step5(tmp_path: Path) -> None:
    config = Step5Config(
        profile="publication",
        radial_order=35,
        time_points=256,
        quadrature_nodes=48,
        closure_tolerance=1e-9,
    )
    result = run_harmonic_kernel(
        tmp_path / "out", _step4_evidence(tmp_path / "step4", config), config
    )
    assert result["manifest"]["status"] == "complete"
    assert result["manifest"]["allowed_next_step"] == 6
    assert result["manifest"]["gates"]["passed"] is True
    assert result["manifest"]["gates"]["selection_rule_support_passed"] is True
    assert result["manifest"]["gates"]["kernel_asymptotic_2pct_passed"] is True
    assert len(result["step4_continuity"]) == 6
