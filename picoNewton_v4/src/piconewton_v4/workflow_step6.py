"""Step 6 workflow: passive standard-linear-solid membrane--cortex interface."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .membrane import (
    ForceLoadMap,
    MembraneAdmissibleDomain,
    SLSParameters,
    WSSLoadMap,
    complex_compliance,
    complex_modulus,
    harmonic_energy_balance,
    periodic_response,
    step_creep_response,
    validate_passivity,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_membrane_config(package_root: Path) -> tuple[SLSParameters, WSSLoadMap, ForceLoadMap]:
    raw = json.loads((package_root / "configs" / "membrane.json").read_text(encoding="utf-8"))
    params = SLSParameters(**raw["sls"])
    wss_map = WSSLoadMap(**raw["wss_mapping"])
    force_map = ForceLoadMap(**raw["force_mapping"])
    params.validate(allow_elastic_limit=False)
    wss_map.validate()
    force_map.validate()
    return params, wss_map, force_map


def _hydrodynamic_source(
    *, package_root: Path, output_root: Path, profile: str, hydrodynamic_root: Path | None
) -> tuple[Path, dict[str, object] | None]:
    if hydrodynamic_root is not None:
        root = hydrodynamic_root.resolve()
        required = root / "six_artery_hydrodynamics.npz"
        if not required.exists():
            raise FileNotFoundError(required)
        return root, None
    from .workflow_step4 import run_step4

    root = output_root / f"_step4_{profile}"
    manifest = run_step4(package_root=package_root, output_root=root, profile=profile)
    if manifest["status"] != "passed":
        raise RuntimeError("Step 4 hydrodynamics did not pass")
    return root, manifest


def _artery_ids(archive: np.lib.npyio.NpzFile) -> list[str]:
    suffix = "_time_cycle"
    return sorted(key[: -len(suffix)] for key in archive.files if key.endswith(suffix))


def _summary_row(artery_id: str, pathway: str, response: dict[str, np.ndarray]) -> dict[str, object]:
    stress = response["equivalent_stress_pa"]
    strain = response["equivalent_areal_strain"]
    tension = response["total_applied_tension_n_m"]
    dissipation = response["dissipation_density_w_m3"]
    return {
        "artery_id": artery_id,
        "pathway": pathway,
        "stress_mean_pa": float(np.mean(stress)),
        "stress_rms_pa": float(np.sqrt(np.mean(stress**2))),
        "stress_peak_abs_pa": float(np.max(np.abs(stress))),
        "strain_mean": float(np.mean(strain)),
        "strain_rms": float(np.sqrt(np.mean(strain**2))),
        "strain_peak_abs": float(np.max(np.abs(strain))),
        "strain_dynamic_range": float(np.ptp(strain)),
        "applied_tension_rms_n_m": float(np.sqrt(np.mean(tension**2))),
        "mean_dissipation_density_w_m3": float(np.mean(dissipation)),
    }


def run_step6(
    *,
    package_root: Path,
    output_root: Path,
    profile: str = "quick",
    hydrodynamic_root: Path | None = None,
) -> dict[str, object]:
    """Map Step 4 WSS and signed force separately through one passive SLS."""

    package_root = package_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    params, wss_map, force_map = load_membrane_config(package_root)
    domain = MembraneAdmissibleDomain()
    hydro_root, generated_step4 = _hydrodynamic_source(
        package_root=package_root,
        output_root=output_root,
        profile=profile,
        hydrodynamic_root=hydrodynamic_root,
    )
    hydro_path = hydro_root / "six_artery_hydrodynamics.npz"

    domain_path = output_root / "admissible_parameter_domain.csv"
    pd.DataFrame(domain.as_rows()).to_csv(domain_path, index=False)

    frequencies_hz = np.concatenate(([0.0], np.geomspace(1e-3, 100.0, 400)))
    omega = 2.0 * np.pi * frequencies_hz
    compliance = complex_compliance(omega, params)
    modulus = complex_modulus(omega, params)
    transfer_path = output_root / "membrane_transfer_function.csv"
    pd.DataFrame(
        {
            "frequency_hz": frequencies_hz,
            "compliance_storage_pa_inv": compliance.real,
            "compliance_loss_pa_inv": -compliance.imag,
            "modulus_storage_pa": modulus.real,
            "modulus_loss_pa": modulus.imag,
            "strain_phase_lag_rad": -np.angle(compliance),
        }
    ).to_csv(transfer_path, index=False)

    summary_rows: list[dict[str, object]] = []
    waveform_rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    minimum_dissipation = np.inf
    maximum_tension_closure = 0.0
    all_finite = True

    with np.load(hydro_path) as archive:
        artery_ids = _artery_ids(archive)
        for artery_id in artery_ids:
            time_cycle = np.asarray(archive[f"{artery_id}_time_cycle"], dtype=float)
            if time_cycle.ndim != 1 or time_cycle.size < 8:
                raise ValueError(f"invalid time cycle for {artery_id}")
            cycle_period_s = 1.0 / 1.2
            dt_s = cycle_period_s / time_cycle.size
            wss_aniso = np.asarray(archive[f"{artery_id}_wss_anisotropic_pa"], dtype=float)
            wss_iso = np.asarray(archive[f"{artery_id}_wss_isotropic_pa"], dtype=float)
            force_aniso = np.asarray(archive[f"{artery_id}_force_signed_anisotropic_n"], dtype=float)
            force_iso = np.asarray(archive[f"{artery_id}_force_signed_isotropic_n"], dtype=float)
            force_increment = np.asarray(
                archive[f"{artery_id}_force_signed_anisotropy_increment_n"], dtype=float
            )
            mapped = {
                "wss_anisotropic": wss_map.map(wss_aniso),
                "wss_isotropic": wss_map.map(wss_iso),
                "wss_anisotropy_increment": wss_map.map(wss_aniso - wss_iso),
                "force_signed_anisotropic": force_map.map(force_aniso),
                "force_signed_isotropic": force_map.map(force_iso),
                "force_signed_anisotropy_increment": force_map.map(force_increment),
            }
            arrays[f"{artery_id}_time_cycle"] = time_cycle
            for pathway, stress in mapped.items():
                response = periodic_response(stress, dt_s=dt_s, params=params)
                summary_rows.append(_summary_row(artery_id, pathway, response))
                for name, values in response.items():
                    arrays[f"{artery_id}_{pathway}_{name}"] = values
                minimum_dissipation = min(
                    minimum_dissipation,
                    float(np.min(response["dissipation_density_w_m3"])),
                )
                closure = np.max(
                    np.abs(
                        response["relaxed_branch_tension_n_m"]
                        + response["maxwell_branch_tension_n_m"]
                        - response["total_applied_tension_n_m"]
                    )
                )
                maximum_tension_closure = max(maximum_tension_closure, float(closure))
                all_finite = all_finite and bool(
                    all(np.all(np.isfinite(value)) for value in response.values())
                )
                for i, phase in enumerate(time_cycle):
                    waveform_rows.append(
                        {
                            "artery_id": artery_id,
                            "pathway": pathway,
                            "time_cycle": float(phase),
                            "equivalent_stress_pa": float(response["equivalent_stress_pa"][i]),
                            "equivalent_areal_strain": float(response["equivalent_areal_strain"][i]),
                            "relaxed_branch_tension_n_m": float(response["relaxed_branch_tension_n_m"][i]),
                            "maxwell_branch_tension_n_m": float(response["maxwell_branch_tension_n_m"][i]),
                            "total_applied_tension_n_m": float(response["total_applied_tension_n_m"][i]),
                            "dissipation_density_w_m3": float(response["dissipation_density_w_m3"][i]),
                        }
                    )

    summary_path = output_root / "six_artery_membrane_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    waveform_path = output_root / "six_artery_membrane_waveforms.csv"
    pd.DataFrame(waveform_rows).to_csv(waveform_path, index=False)
    states_path = output_root / "six_artery_membrane_states.npz"
    np.savez_compressed(states_path, **arrays)

    zero = periodic_response(np.zeros(128), dt_s=0.01, params=params)
    time = np.array([0.0, 50.0 * params.creep_time_s])
    creep = step_creep_response(time, stress_step_pa=1.0, params=params)
    elastic = SLSParameters(
        instantaneous_modulus_pa=params.instantaneous_modulus_pa,
        relaxed_modulus_pa=params.instantaneous_modulus_pa,
        stress_relaxation_time_s=params.stress_relaxation_time_s,
        thickness_m=params.thickness_m,
    )
    elastic_stress = np.sin(2.0 * np.pi * np.arange(256) / 256.0)
    elastic_response = periodic_response(elastic_stress, dt_s=0.01, params=elastic)
    elastic_error = float(
        np.max(
            np.abs(
                elastic_response["equivalent_areal_strain"]
                - elastic_stress / params.instantaneous_modulus_pa
            )
        )
    )
    low_compliance = float(np.real(complex_compliance(1e-12, params)))
    high_compliance = float(np.real(complex_compliance(1e12, params)))
    passivity = validate_passivity(params)
    energy = harmonic_energy_balance(
        stress_amplitude_pa=1.0,
        frequency_hz=1.2,
        params=params,
    )

    validation = {
        "all_six_arteries_present": len(artery_ids) == 6,
        "all_outputs_finite": bool(all_finite),
        "default_parameters_in_primary_domain": domain.contains(params),
        "force_mapping_10_pn_to_pa": float(force_map.map(np.array([10e-12]))[0]),
        "wss_mapping_1_pa_to_pa": float(wss_map.map(np.array([1.0]))[0]),
        "zero_input_max_abs_strain": float(np.max(np.abs(zero["equivalent_areal_strain"]))),
        "step_initial_strain_error": float(abs(creep[0] - 1.0 / params.instantaneous_modulus_pa)),
        "step_relaxed_strain_error": float(abs(creep[-1] - 1.0 / params.relaxed_modulus_pa)),
        "low_frequency_compliance_error": float(abs(low_compliance - 1.0 / params.relaxed_modulus_pa)),
        "high_frequency_compliance_error": float(abs(high_compliance - 1.0 / params.instantaneous_modulus_pa)),
        "elastic_limit_max_abs_error": elastic_error,
        "minimum_dissipation_density_w_m3": float(minimum_dissipation),
        "maximum_branch_tension_closure_n_m": float(maximum_tension_closure),
        "passivity": passivity,
        "harmonic_energy_balance": dict(energy),
        "primary_force_input": "force_signed_anisotropic_n",
        "magnitude_force_exposure_used_as_signed_load": False,
        "piezo1_coupling_executed": False,
    }
    validation["passed"] = bool(
        validation["all_six_arteries_present"]
        and validation["all_outputs_finite"]
        and validation["default_parameters_in_primary_domain"]
        and abs(validation["force_mapping_10_pn_to_pa"] - 0.1) < 1e-14
        and abs(validation["wss_mapping_1_pa_to_pa"] - 1.0) < 1e-14
        and validation["zero_input_max_abs_strain"] < 1e-15
        and validation["step_initial_strain_error"] < 1e-15
        and validation["step_relaxed_strain_error"] < 1e-12
        and validation["low_frequency_compliance_error"] < 1e-12
        and validation["high_frequency_compliance_error"] < 1e-12
        and validation["elastic_limit_max_abs_error"] < 1e-15
        and validation["minimum_dissipation_density_w_m3"] >= -1e-15
        and validation["maximum_branch_tension_closure_n_m"] < 1e-18
        and passivity["passed"]
        and energy["relative_error"] < 1e-12
        and validation["magnitude_force_exposure_used_as_signed_load"] is False
        and validation["piezo1_coupling_executed"] is False
    )
    validation_path = output_root / "step6_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    produced = [
        domain_path,
        transfer_path,
        summary_path,
        waveform_path,
        states_path,
        validation_path,
    ]
    manifest = {
        "step": 6,
        "status": "passed" if validation["passed"] else "failed",
        "profile": profile,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "sls_parameters": asdict(params),
        "derived_parameters": {
            "maxwell_modulus_pa": params.maxwell_modulus_pa,
            "viscosity_pa_s": params.viscosity_pa_s,
            "creep_time_s": params.creep_time_s,
        },
        "wss_mapping": asdict(wss_map),
        "force_mapping": asdict(force_map),
        "hydrodynamic_source": {
            "path": str(hydro_path),
            "sha256": _sha256(hydro_path),
            "generated_in_step6": generated_step4 is not None,
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in produced
        },
        "validation": validation,
        "claim_boundary": (
            "Step 6 derives and validates a passive reduced-order SLS interface and maps WSS "
            "and signed force separately to equivalent areal strain. It does not establish a "
            "unique endothelial load-transfer law and does not execute Piezo1 coupling."
        ),
    }
    manifest_path = output_root / "step6_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
