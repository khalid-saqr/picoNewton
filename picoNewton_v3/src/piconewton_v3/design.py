"""Physiological Sobol design and resumable coverage execution."""
from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr

from .model import (
    ArteryCase, EndothelialControlVolume, FluidProperties, HydrodynamicConfig,
    SensorConfig, V2_ARTERY_CASES, compute_hydrodynamics, lamb_work,
    periodic_sensor_solution, rms_difference, signal_metrics, wss_work,
)
from .study_io import StudyStore
from .workflow_common import (
    DEFAULT_SEED, NUMERICAL_SENSOR_UNCERTAINTY, _phase_distance, _rms,
    _sensor_from_work, run_hydrodynamic_cases,
)

def generate_sobol_design(
    samples: int,
    *,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    if samples <= 0 or samples & (samples - 1):
        raise ValueError("Sobol samples must be a positive power of two")
    sampler = qmc.Sobol(d=8, scramble=True, seed=seed)
    unit = sampler.random_base2(int(np.log2(samples)))
    rows: list[dict[str, float | int]] = []
    for index, point in enumerate(unit):
        rows.append(
            {
                "sample_id": index,
                "heart_rate_bpm": 60.0 + 40.0 * point[0],
                "density_kg_m3": 1040.0 + 40.0 * point[1],
                "dynamic_viscosity_pa_s": 3e-3 + 1e-3 * point[2],
                "beta": -0.1 + 0.2 * point[3],
                "gamma": -0.1 + 0.2 * point[4],
                "delta": 0.9 + 0.2 * point[5],
                "coupling_length_m": 10 ** (-10.0 + 2.0 * point[6]),
                "relaxation_time_s": 10 ** (-3.0 + 4.0 * point[7]),
            }
        )
    design = pd.DataFrame(rows)
    margin = design["delta"] - ((design["beta"] + design["gamma"]) / 2.0) ** 2
    if not (margin > 0).all():
        raise RuntimeError("Sobol design contains inadmissible constitutive samples")
    return design

def generate_physiological_design(
    artery_ranges: pd.DataFrame,
    cases: Sequence[ArteryCase],
    samples: int,
    *,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate a site-stratified Sobol coverage design.

    The design is a computational coverage set, not a population probability
    model. Samples are distributed as evenly as possible across the six sites.
    """
    if samples <= 0 or samples & (samples - 1):
        raise ValueError("samples must be a positive power of two")
    primary = artery_ranges[artery_ranges["range_mode"] == "primary"].set_index("artery_id")
    case_map = {case.artery_id: case for case in cases}
    missing = set(case_map) - set(primary.index)
    if missing:
        raise ValueError(f"missing artery ranges: {sorted(missing)}")
    sampler = qmc.Sobol(d=8, scramble=True, seed=seed)
    unit = sampler.random_base2(int(np.log2(samples)))
    artery_ids = [case.artery_id for case in cases]
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(unit):
        artery_id = artery_ids[index % len(artery_ids)]
        ranges = primary.loc[artery_id]
        base = case_map[artery_id]
        radius = ranges.radius_min_m + point[0] * (ranges.radius_max_m - ranges.radius_min_m)
        heart_rate = 60.0 + 40.0 * point[1]
        density = 1040.0 + 40.0 * point[2]
        dynamic_viscosity = 3e-3 + 1e-3 * point[3]
        beta = -0.1 + 0.2 * point[4]
        gamma = -0.1 + 0.2 * point[5]
        delta = 0.9 + 0.2 * point[6]
        # Primary analysis preserves G0 exactly. The final Sobol dimension is
        # retained as a deterministic phase-space coordinate for auditability.
        rows.append(
            {
                "sample_id": f"phys_{index:05d}",
                "artery_id": artery_id,
                "artery_name": base.name,
                "radius_m": radius,
                "heart_rate_bpm": heart_rate,
                "density_kg_m3": density,
                "dynamic_viscosity_pa_s": dynamic_viscosity,
                "kinematic_viscosity_m2_s": dynamic_viscosity / density,
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
                "pressure_gradient_scale_pa_per_m": base.pressure_gradient_scale_pa_per_m,
                "audit_coordinate": point[7],
                "random_seed": seed,
                "interpretation": "coverage sample; not population probability",
            }
        )
    design = pd.DataFrame(rows)
    margin = design["delta"] - ((design["beta"] + design["gamma"]) / 2.0) ** 2
    if not (margin > 0).all():
        raise RuntimeError("coverage design contains inadmissible constitutive samples")
    return design


def run_physiological_coverage(
    design: pd.DataFrame,
    base_cases: Sequence[ArteryCase],
    numerical_profile: HydrodynamicConfig,
    sensor: SensorConfig = SensorConfig(),
    endothelium: EndothelialControlVolume = EndothelialControlVolume(),
    *,
    coupling_length_m: float = 1e-9,
    wss_activation_volume_m3: float = 1e-22,
    checkpoint_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the physiological coverage design with per-sample checkpoints.

    The complete coverage dataset stores summary metrics and DC--h12 spectra for
    every sample. Full near-wall fields are retained by the nominal six-artery
    run and selected validation anchors, avoiding an unnecessarily large field
    archive while preserving every value used in coverage figures and gates.
    """
    case_map = {case.artery_id: case for case in base_cases}
    checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint is not None:
        checkpoint.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    for record in design.to_dict("records"):
        sample_id = str(record["sample_id"])
        summary_path = checkpoint / f"{sample_id}_summary.json" if checkpoint else None
        spectrum_path = checkpoint / f"{sample_id}_spectrum.csv" if checkpoint else None
        if summary_path is not None and summary_path.exists() and spectrum_path is not None and spectrum_path.exists():
            summary_rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
            spectrum_rows.extend(pd.read_csv(spectrum_path).to_dict("records"))
            continue

        base = case_map[str(record["artery_id"])]
        sample_case = ArteryCase(
            artery_id=sample_id,
            name=f"{base.name} [{sample_id}]",
            radius_m=float(record["radius_m"]),
            pressure_gradient_scale_pa_per_m=float(record["pressure_gradient_scale_pa_per_m"]),
            harmonic_coefficients=base.harmonic_coefficients,
        )
        fluid = FluidProperties(
            density_kg_m3=float(record["density_kg_m3"]),
            kinematic_viscosity_m2_s=float(record["kinematic_viscosity_m2_s"]),
            fundamental_frequency_hz=float(record["heart_rate_bpm"]) / 60.0,
        )
        hydro_config = HydrodynamicConfig(
            radial_order=numerical_profile.radial_order,
            time_points=numerical_profile.time_points,
            quadrature_nodes=numerical_profile.quadrature_nodes,
            beta=float(record["beta"]),
            gamma=float(record["gamma"]),
            delta=float(record["delta"]),
            mode="verified",
        )
        hydro = compute_hydrodynamics(sample_case, hydro_config, fluid, endothelium)
        force = np.asarray(hydro["force_signed_n"])
        exposure = np.asarray(hydro["force_exposure_n"])
        shear = np.asarray(hydro["wall_shear_pa"])
        psi_l = lamb_work(force, coupling_length_m, sensor.temperature_k)
        psi_w = wss_work(shear, wss_activation_volume_m3, sensor.temperature_k)
        p_l, _ = periodic_sensor_solution(psi_l, fluid.fundamental_frequency_hz, sensor)
        p_w, _ = periodic_sensor_solution(psi_w, fluid.fundamental_frequency_hz, sensor)
        p_parallel, periodic_residual = periodic_sensor_solution(
            psi_l + psi_w, fluid.fundamental_frequency_hz, sensor
        )
        force_metrics = signal_metrics(force)
        summary = {
            **record,
            "alpha": float(hydro["alpha"]),
            "force_signed_mean_n": float(np.mean(force)),
            "force_signed_rms_n": _rms(force),
            "force_peak_abs_n": float(np.max(np.abs(force))),
            "force_exposure_mean_n": float(np.mean(exposure)),
            "wall_shear_rms_pa": _rms(shear),
            "force_high_harmonic_power_fraction": force_metrics["high_harmonic_power_fraction"],
            "p_lamb_mean": float(np.mean(p_l)),
            "p_wss_mean": float(np.mean(p_w)),
            "p_parallel_mean": float(np.mean(p_parallel)),
            "p_parallel_dynamic_range": float(np.ptp(p_parallel)),
            "effect_parallel_vs_wss": rms_difference(p_parallel, p_w),
            "periodic_residual": periodic_residual,
        }
        summary_rows.append(summary)

        sample_spectra: list[dict[str, Any]] = []
        for signal_name, values in (("force_signed_n", force), ("wall_shear_pa", shear)):
            coefficients = np.fft.rfft(values) / len(values)
            for harmonic in range(min(12, len(coefficients) - 1) + 1):
                coefficient = coefficients[harmonic]
                sample_spectra.append(
                    {
                        "sample_id": sample_id,
                        "artery_id": record["artery_id"],
                        "signal": signal_name,
                        "harmonic": harmonic,
                        "amplitude": float(np.abs(coefficient)),
                        "phase_rad": float(np.angle(coefficient)),
                        "power": float(np.abs(coefficient) ** 2),
                    }
                )
        spectrum_rows.extend(sample_spectra)
        if summary_path is not None and spectrum_path is not None:
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            pd.DataFrame(sample_spectra).to_csv(spectrum_path, index=False)
        # Barycentric interpolators and spectral matrices can retain temporary
        # reference cycles in long coverage sweeps. Explicit collection keeps
        # runtime and memory stable across thousands of checkpointed samples.
        del hydro, force, exposure, shear, p_l, p_w, p_parallel
        gc.collect()
    return pd.DataFrame(summary_rows), pd.DataFrame(spectrum_rows)
