from pathlib import Path
import json

import numpy as np

from piconewton_v4.workflow_step6 import run_step6


def test_step6_with_six_artery_fixture(tmp_path: Path):
    package_root = tmp_path / "package"
    (package_root / "configs").mkdir(parents=True)
    (package_root / "configs" / "membrane.json").write_text(
        json.dumps(
            {
                "sls": {
                    "instantaneous_modulus_pa": 2500.0,
                    "relaxed_modulus_pa": 1000.0,
                    "stress_relaxation_time_s": 0.25,
                    "thickness_m": 3.5e-7,
                },
                "wss_mapping": {"transfer_fraction": 1.0},
                "force_mapping": {
                    "effective_area_m2": 1e-10,
                    "transfer_fraction": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    hydro = tmp_path / "hydro"
    hydro.mkdir()
    n = 128
    phase = np.arange(n) / n
    arrays = {}
    for index, artery in enumerate(
        [
            "aortic_root",
            "thoracic_aorta",
            "femoral",
            "carotid",
            "iliac",
            "brachial",
        ],
        start=1,
    ):
        wave = np.sin(2 * np.pi * phase)
        arrays[f"{artery}_time_cycle"] = phase
        arrays[f"{artery}_wss_anisotropic_pa"] = index * wave
        arrays[f"{artery}_wss_isotropic_pa"] = 0.9 * index * wave
        arrays[f"{artery}_force_signed_anisotropic_n"] = index * 1e-12 * wave
        arrays[f"{artery}_force_signed_isotropic_n"] = 0.8 * index * 1e-12 * wave
        arrays[f"{artery}_force_signed_anisotropy_increment_n"] = (
            0.2 * index * 1e-12 * wave
        )
    np.savez_compressed(hydro / "six_artery_hydrodynamics.npz", **arrays)

    manifest = run_step6(
        package_root=package_root,
        output_root=tmp_path / "output",
        hydrodynamic_root=hydro,
    )
    assert manifest["status"] == "passed"
    assert manifest["validation"]["piezo1_coupling_executed"] is False
    assert (
        manifest["validation"]["magnitude_force_exposure_used_as_signed_load"]
        is False
    )
