"""Step 4 six-artery hydrodynamic workflow and export."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .hydrodynamics import (
    compute_decomposition,
    convergence_comparison,
    isotropic_validation,
    spectral_amplitudes,
    spectral_tail_power_fraction,
    summarize_decomposition,
)
from .types import HydrodynamicConfig, load_artery_cases


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_profile(package_root: Path, profile: str) -> HydrodynamicConfig:
    config_path = package_root / "configs" / f"{profile}.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    hydro = raw["hydrodynamics"]
    return HydrodynamicConfig(
        radial_order=int(hydro["radial_order"]),
        time_points=int(hydro["time_points"]),
        near_wall_nodes=int(hydro["near_wall_nodes"]),
        beta=float(hydro.get("beta", 0.1)),
        gamma=float(hydro.get("gamma", 0.1)),
        delta=float(hydro.get("delta", 1.0)),
    )


def run_step4(
    *,
    package_root: Path,
    output_root: Path,
    profile: str = "quick",
) -> dict[str, object]:
    package_root = package_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_profile(package_root, profile)
    cases = load_artery_cases(package_root / "data" / "ground_truth_arteries.csv")

    decompositions = [compute_decomposition(case, config) for case in cases]
    signed_tail_h12 = max(
        spectral_tail_power_fraction(np.asarray(item["force_signed_anisotropic_n"]), 12)
        for item in decompositions
    )
    wss_tail_h6 = max(
        spectral_tail_power_fraction(np.asarray(item["wss_anisotropic_pa"]), 6)
        for item in decompositions
    )
    summary = pd.DataFrame(summarize_decomposition(item) for item in decompositions)
    summary_path = output_root / "six_artery_hydrodynamic_summary.csv"
    summary.to_csv(summary_path, index=False)

    waveform_rows = []
    spectrum_rows = []
    arrays: dict[str, np.ndarray] = {}
    for item in decompositions:
        artery_id = str(item["artery_id"])
        time_cycle = np.asarray(item["time_cycle"], dtype=float)
        series = {
            "wss_anisotropic_pa": np.asarray(item["wss_anisotropic_pa"], dtype=float),
            "wss_isotropic_pa": np.asarray(item["wss_isotropic_pa"], dtype=float),
            "force_signed_anisotropic_n": np.asarray(item["force_signed_anisotropic_n"], dtype=float),
            "force_signed_isotropic_n": np.asarray(item["force_signed_isotropic_n"], dtype=float),
            "force_signed_anisotropy_increment_n": np.asarray(item["force_signed_anisotropy_increment_n"], dtype=float),
            "force_exposure_anisotropic_n": np.asarray(item["force_exposure_anisotropic_n"], dtype=float),
            "force_exposure_isotropic_n": np.asarray(item["force_exposure_isotropic_n"], dtype=float),
            "force_exposure_anisotropy_increment_n": np.asarray(item["force_exposure_anisotropy_increment_n"], dtype=float),
        }
        arrays[f"{artery_id}_time_cycle"] = time_cycle
        for signal_name, values in series.items():
            arrays[f"{artery_id}_{signal_name}"] = values
        for i, t in enumerate(time_cycle):
            waveform_rows.append({
                "artery_id": artery_id,
                "time_cycle": float(t),
                **{key: float(value[i]) for key, value in series.items()},
            })
        for signal_name, values in series.items():
            amplitudes = spectral_amplitudes(values, max_harmonic=12)
            for harmonic, amplitude in enumerate(amplitudes):
                spectrum_rows.append({
                    "artery_id": artery_id,
                    "signal": signal_name,
                    "harmonic": harmonic,
                    "amplitude": float(amplitude),
                })

    waveform_path = output_root / "six_artery_hydrodynamic_waveforms.csv"
    pd.DataFrame(waveform_rows).to_csv(waveform_path, index=False)
    spectrum_path = output_root / "six_artery_hydrodynamic_spectra.csv"
    pd.DataFrame(spectrum_rows).to_csv(spectrum_path, index=False)
    npz_path = output_root / "six_artery_hydrodynamics.npz"
    np.savez_compressed(npz_path, **arrays)

    iso_rows = isotropic_validation(
        alpha_values=tuple(case.womersley_alpha_reference for case in cases),
        radial_order=max(config.radial_order, 80 if profile == "quick" else 150),
    )
    iso_path = output_root / "isotropic_womersley_validation.csv"
    pd.DataFrame(iso_rows).to_csv(iso_path, index=False)

    thoracic = next(case for case in cases if case.artery_id == "thoracic_aorta")
    if profile == "publication":
        convergence_coarse = HydrodynamicConfig(
            radial_order=150,
            time_points=config.time_points,
            near_wall_nodes=config.near_wall_nodes,
            beta=config.beta,
            gamma=config.gamma,
            delta=config.delta,
        )
        fine_order = 180
    else:
        convergence_coarse = config
        fine_order = 64
    convergence = convergence_comparison(thoracic, convergence_coarse, fine_order)
    convergence_path = output_root / "hydrodynamic_convergence.json"
    convergence_path.write_text(json.dumps(convergence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    validation = {
        "all_six_arteries_present": len(summary) == 6,
        "alpha_max_abs_error": float(summary["alpha_abs_error"].max()),
        "max_isotropic_linf_error": float(max(float(row["linf_error"]) for row in iso_rows)),
        "max_backward_residual": float(summary["max_normalized_backward_residual"].max()),
        "max_gromeka_lamb_closure_relative_error": float(summary["gromeka_lamb_closure_relative_error"].max()),
        "max_signed_force_power_above_h12": float(signed_tail_h12),
        "max_wss_power_above_h6": float(wss_tail_h6),
        "all_outputs_finite": bool(np.isfinite(summary.select_dtypes(include=[float, int]).to_numpy()).all()),
        "force_exposure_nonnegative": bool(
            all(np.min(np.asarray(item["force_exposure_anisotropic_n"])) >= 0.0 for item in decompositions)
        ),
        "convergence": convergence,
    }
    if profile == "publication":
        validation["passed"] = bool(
            validation["all_six_arteries_present"]
            and validation["alpha_max_abs_error"] < 0.02
            and validation["max_isotropic_linf_error"] < 1e-8
            and validation["max_backward_residual"] < 1e-13
            and validation["max_gromeka_lamb_closure_relative_error"] < 1e-10
            and validation["max_signed_force_power_above_h12"] < 1e-12
            and validation["max_wss_power_above_h6"] < 1e-12
            and validation["all_outputs_finite"]
            and validation["force_exposure_nonnegative"]
            and convergence["force_signed_n_relative_l2"] < 1e-4
            and convergence["force_exposure_n_relative_l2"] < 1e-4
            and convergence["wall_shear_pa_relative_l2"] < 1e-6
        )
    else:
        validation["passed"] = bool(
            validation["all_six_arteries_present"]
            and validation["alpha_max_abs_error"] < 0.02
            and validation["max_isotropic_linf_error"] < 1e-7
            and validation["max_backward_residual"] < 1e-12
            and validation["max_gromeka_lamb_closure_relative_error"] < 1e-8
            and validation["max_signed_force_power_above_h12"] < 1e-10
            and validation["max_wss_power_above_h6"] < 1e-10
            and validation["all_outputs_finite"]
            and validation["force_exposure_nonnegative"]
            and convergence["force_signed_n_relative_l2"] < 5e-3
            and convergence["force_exposure_n_relative_l2"] < 5e-3
            and convergence["wall_shear_pa_relative_l2"] < 5e-4
        )
    validation_path = output_root / "step4_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    produced = [
        summary_path,
        waveform_path,
        spectrum_path,
        npz_path,
        iso_path,
        convergence_path,
        validation_path,
    ]
    manifest = {
        "step": 4,
        "status": "passed" if validation["passed"] else "failed",
        "profile": profile,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "hydrodynamic_config": asdict(config),
        "inputs": {
            "arteries": str(package_root / "data" / "ground_truth_arteries.csv"),
            "arteries_sha256": _sha256(package_root / "data" / "ground_truth_arteries.csv"),
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in produced
        },
        "validation": validation,
        "claim_boundary": (
            "Hydrodynamic reproduction only. No membrane loading, Piezo1 gating, "
            "or biological inference is performed in Step 4."
        ),
    }
    manifest_path = output_root / "step4_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
