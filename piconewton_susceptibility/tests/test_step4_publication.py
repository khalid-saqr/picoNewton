import hashlib
import json
from pathlib import Path

import numpy as np
from piconewton_v3 import (
    V2_ARTERY_CASES,
    EndothelialControlVolume,
    FluidProperties,
    HydrodynamicConfig,
    compute_hydrodynamics,
)

from piconewton_susceptibility.perturbation import Step4Config, run_perturbative_hierarchy


def _step3_gate(root: Path) -> Path:
    root.mkdir(parents=True)
    arrays = {}
    for case in V2_ARTERY_CASES:
        anisotropic = compute_hydrodynamics(
            case,
            HydrodynamicConfig(
                radial_order=60,
                time_points=512,
                quadrature_nodes=96,
                beta=0.1,
                gamma=0.1,
                delta=1.0,
                mode="verified",
            ),
            FluidProperties(),
            EndothelialControlVolume(),
        )
        isotropic = compute_hydrodynamics(
            case,
            HydrodynamicConfig(
                radial_order=60,
                time_points=512,
                quadrature_nodes=96,
                beta=0.0,
                gamma=0.0,
                delta=1.0,
                mode="verified",
            ),
            FluidProperties(),
            EndothelialControlVolume(),
        )
        arrays[f"{case.artery_id}__signed_anisotropic_n"] = anisotropic["force_signed_n"]
        arrays[f"{case.artery_id}__signed_isotropic_n"] = isotropic["force_signed_n"]
    archive = root / "six_artery_waveforms.npz"
    np.savez_compressed(archive, **arrays)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (root / "step3_gate.json").write_text('{"passed":true}', encoding="utf-8")
    (root / "step3_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "allowed_next_step": 4,
                "files": {archive.name: {"sha256": digest}},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_reduced_resolution_publication_profile_closes_step4(tmp_path: Path) -> None:
    config = Step4Config(
        profile="publication",
        radial_order=60,
        time_points=512,
        quadrature_nodes=96,
        epsilon_values=(0.005, 0.01, 0.02, 0.04, 0.08, 0.10),
        slope_epsilon_max=0.04,
        parity_epsilon=0.04,
        minimum_valid_epsilon=0.04,
    )
    result = run_perturbative_hierarchy(
        tmp_path / "out", _step3_gate(tmp_path / "step3"), config
    )
    assert result["manifest"]["status"] == "complete"
    assert result["manifest"]["allowed_next_step"] == 5
    assert result["manifest"]["gates"]["passed"] is True
    assert result["manifest"]["gates"]["step3_waveform_continuity_passed"] is True
