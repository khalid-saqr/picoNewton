"""Predeclared mechanosensory controls and WSS surrogate fitting."""
from __future__ import annotations

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

def run_nominal_controls(
    cases: Sequence[ArteryCase],
    config: HydrodynamicConfig,
    sensor: SensorConfig = SensorConfig(),
    fluid: FluidProperties = FluidProperties(),
    endothelium: EndothelialControlVolume = EndothelialControlVolume(),
    *,
    coupling_length_m: float = 1e-9,
    wss_activation_volume_m3: float = 1e-22,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    """Execute C0-C12 and return tidy control and waveform tables."""
    verified = run_hydrodynamic_cases(cases, config, fluid, endothelium)
    isotropic_config = HydrodynamicConfig(
        radial_order=config.radial_order,
        time_points=config.time_points,
        quadrature_nodes=config.quadrature_nodes,
        beta=0.0,
        gamma=0.0,
        delta=1.0,
        mode="verified",
    )
    isotropic = run_hydrodynamic_cases(cases, isotropic_config, fluid, endothelium)

    low_harmonic = {
        case.artery_id: compute_hydrodynamics(
            case,
            config,
            fluid,
            endothelium,
            harmonics_retained=2,
        )
        for case in cases
    }
    rng = np.random.default_rng(seed + 10)
    scrambled_phase = rng.uniform(-np.pi, np.pi, 6)
    scrambled_phase[0] = 0.0
    scrambled = {
        case.artery_id: compute_hydrodynamics(
            case,
            config,
            fluid,
            endothelium,
            phases_rad=scrambled_phase,
        )
        for case in cases
    }

    control_rows: list[dict[str, Any]] = []
    waveform_rows: list[dict[str, Any]] = []
    for case in cases:
        base = verified[case.artery_id]
        iso = isotropic[case.artery_id]
        low = low_harmonic[case.artery_id]
        phase = scrambled[case.artery_id]

        force = np.asarray(base["force_signed_n"])
        force_iso = np.asarray(iso["force_signed_n"])
        force_low = np.asarray(low["force_signed_n"])
        shear = np.asarray(base["wall_shear_pa"])
        psi_l = lamb_work(
            force,
            coupling_length_m,
            sensor.temperature_k,
            mode="signed",
        )
        psi_w = wss_work(
            shear,
            wss_activation_volume_m3,
            sensor.temperature_k,
            mode="signed",
        )

        zero, rz = _sensor_from_work(np.zeros_like(psi_l), fluid, sensor)
        wss_only, rw = _sensor_from_work(psi_w, fluid, sensor)
        lamb_only, rl = _sensor_from_work(psi_l, fluid, sensor)
        parallel, rp = _sensor_from_work(psi_w + psi_l, fluid, sensor)
        anisotropy_excess, ra = _sensor_from_work(
            lamb_work(
                force - force_iso,
                coupling_length_m,
                sensor.temperature_k,
                mode="signed",
            ),
            fluid,
            sensor,
        )
        isotropic_lamb, ri = _sensor_from_work(
            lamb_work(force_iso, coupling_length_m, sensor.temperature_k),
            fluid,
            sensor,
        )
        low_lamb, rlow = _sensor_from_work(
            lamb_work(force_low, coupling_length_m, sensor.temperature_k),
            fluid,
            sensor,
        )
        scrambled_lamb, rs = _sensor_from_work(
            lamb_work(
                np.asarray(phase["force_signed_n"]),
                coupling_length_m,
                sensor.temperature_k,
            ),
            fluid,
            sensor,
        )

        t = np.asarray(base["time_cycle"])
        peak_phase = t[int(np.argmax(np.abs(force)))]
        rms_matched_sine = np.sqrt(2.0) * _rms(force) * np.sin(
            2.0 * np.pi * (t - peak_phase)
        )
        sine_lamb, rsine = _sensor_from_work(
            lamb_work(
                rms_matched_sine,
                coupling_length_m,
                sensor.temperature_k,
            ),
            fluid,
            sensor,
        )

        # C9 is deliberately a waveform-information control, not a physical law.
        matched_wss_work = shear / max(_rms(shear), 1e-30) * _rms(psi_l)
        matched_wss, rmatch = _sensor_from_work(matched_wss_work, fluid, sensor)
        reverse_direction, rneg = _sensor_from_work(
            lamb_work(
                force,
                coupling_length_m,
                sensor.temperature_k,
                signed_sensitivity=-1.0,
            ),
            fluid,
            sensor,
        )
        magnitude_lamb, rmag = _sensor_from_work(
            lamb_work(
                force,
                coupling_length_m,
                sensor.temperature_k,
                mode="magnitude",
            ),
            fluid,
            sensor,
        )

        controls = {
            "C0_zero": (zero, rz),
            "C1_WSS": (wss_only, rw),
            "C2_Lamb": (lamb_only, rl),
            "C3_parallel": (parallel, rp),
            "C4_anisotropy_excess": (anisotropy_excess, ra),
            "C5_isotropic_Lamb": (isotropic_lamb, ri),
            "C6_low_harmonic": (low_lamb, rlow),
            "C7_phase_scrambled": (scrambled_lamb, rs),
            "C8_rms_matched_sinusoid": (sine_lamb, rsine),
            "C9_amplitude_matched_WSS": (matched_wss, rmatch),
            "C11_reverse_direction": (reverse_direction, rneg),
            "C12_magnitude": (magnitude_lamb, rmag),
        }
        for control_id, (probability, residual) in controls.items():
            control_rows.append(
                {
                    "artery_id": case.artery_id,
                    "artery_name": case.name,
                    "control_id": control_id,
                    **signal_metrics(probability),
                    "periodic_residual": residual,
                }
            )

        for i, cycle in enumerate(t):
            waveform_rows.append(
                {
                    "artery_id": case.artery_id,
                    "artery_name": case.name,
                    "time_cycle": cycle,
                    "time_s": np.asarray(base["time_s"])[i],
                    "force_signed_n": force[i],
                    "force_exposure_n": np.asarray(base["force_exposure_n"])[i],
                    "force_isotropic_n": force_iso[i],
                    "force_low_harmonic_n": force_low[i],
                    "wall_shear_pa": shear[i],
                    "p_WSS": wss_only[i],
                    "p_Lamb": lamb_only[i],
                    "p_parallel": parallel[i],
                    "p_reverse_direction": reverse_direction[i],
                    "p_magnitude": magnitude_lamb[i],
                }
            )
    return pd.DataFrame(control_rows), pd.DataFrame(waveform_rows), verified


def run_parameter_grid(
    cases: Sequence[ArteryCase],
    hydrodynamics: dict[str, dict[str, Any]],
    fluid: FluidProperties,
    *,
    coupling_lengths_m: Sequence[float],
    relaxation_times_s: Sequence[float],
    basal_probability: float = 0.01,
    transition_fraction: float = 0.5,
    temperature_k: float = 310.15,
    wss_activation_volume_m3: float = 1e-22,
    numerical_uncertainty: float = NUMERICAL_SENSOR_UNCERTAINTY,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in cases:
        base = hydrodynamics[case.artery_id]
        force = np.asarray(base["force_signed_n"])
        shear = np.asarray(base["wall_shear_pa"])
        # Isotropic and low-harmonic controls are recomputed at the same numerical profile.
        template = HydrodynamicConfig(
            radial_order=int(base["radial_order"]),
            time_points=int(base["time_points"]),
            quadrature_nodes=int(base["quadrature_nodes"]),
            beta=0.0,
            gamma=0.0,
            delta=1.0,
            mode="verified",
        )
        iso = compute_hydrodynamics(case, template, fluid)
        full_config = HydrodynamicConfig(
            radial_order=int(base["radial_order"]),
            time_points=int(base["time_points"]),
            quadrature_nodes=int(base["quadrature_nodes"]),
            beta=float(base.get("beta", 0.1)),
            gamma=float(base.get("gamma", 0.1)),
            delta=float(base.get("delta", 1.0)),
            mode="verified",
        )
        low = compute_hydrodynamics(case, full_config, fluid, harmonics_retained=2)
        force_iso = np.asarray(iso["force_signed_n"])
        force_low = np.asarray(low["force_signed_n"])

        for coupling_length in coupling_lengths_m:
            for relaxation_time in relaxation_times_s:
                sensor = SensorConfig(
                    basal_probability=basal_probability,
                    relaxation_time_s=float(relaxation_time),
                    transition_fraction=transition_fraction,
                    temperature_k=temperature_k,
                )
                psi_l = lamb_work(force, coupling_length, temperature_k)
                psi_w = wss_work(shear, wss_activation_volume_m3, temperature_k)
                p_w, _ = _sensor_from_work(psi_w, fluid, sensor)
                p_parallel, _ = _sensor_from_work(psi_w + psi_l, fluid, sensor)
                p_l, _ = _sensor_from_work(psi_l, fluid, sensor)
                p_low, _ = _sensor_from_work(
                    lamb_work(force_low, coupling_length, temperature_k), fluid, sensor
                )
                p_iso, _ = _sensor_from_work(
                    lamb_work(force_iso, coupling_length, temperature_k), fluid, sensor
                )
                p_reverse, _ = _sensor_from_work(
                    lamb_work(
                        force,
                        coupling_length,
                        temperature_k,
                        signed_sensitivity=-1.0,
                    ),
                    fluid,
                    sensor,
                )

                effect = rms_difference(p_parallel, p_w)
                dynamic_range = max(signal_metrics(p_parallel)["dynamic_range"], 1e-12)
                high_abs = rms_difference(p_l, p_low)
                high_rel = high_abs / max(_rms(p_l - np.mean(p_l)), 1e-12)
                anisotropy = rms_difference(p_l, p_iso)
                direction = rms_difference(p_l, p_reverse)
                phase = _phase_distance(
                    signal_metrics(p_l)["peak_phase_cycle"],
                    signal_metrics(p_reverse)["peak_phase_cycle"],
                )
                rows.append(
                    {
                        "artery_id": case.artery_id,
                        "artery_name": case.name,
                        "coupling_length_m": coupling_length,
                        "coupling_length_nm": coupling_length * 1e9,
                        "relaxation_time_s": relaxation_time,
                        "Omega": 2.0
                        * np.pi
                        * fluid.fundamental_frequency_hz
                        * relaxation_time,
                        "Lambda_RMS": _rms(psi_l),
                        "effect_parallel_vs_WSS": effect,
                        "effect_fraction_dynamic_range": effect / dynamic_range,
                        "passes_E1_point": bool(
                            effect >= 0.005
                            and effect >= 10.0 * numerical_uncertainty
                        ),
                        "high_harmonic_abs": high_abs,
                        "high_harmonic_rel": high_rel,
                        "passes_E5_point": bool(high_abs >= 0.002 and high_rel >= 0.2),
                        "anisotropy_effect": anisotropy,
                        "passes_E6_point": bool(
                            anisotropy >= 0.002
                            and anisotropy >= 10.0 * numerical_uncertainty
                        ),
                        "direction_effect": direction,
                        "direction_phase_difference": phase,
                        "passes_E4_point": bool(direction >= 0.005 or phase >= 0.02),
                    }
                )
    return pd.DataFrame(rows)


def fit_wss_surrogate(
    cases: Sequence[ArteryCase],
    hydrodynamics: dict[str, dict[str, Any]],
    fluid: FluidProperties,
    sensor: SensorConfig,
    *,
    coupling_length_m: float = 1e-9,
    wss_activation_volume_m3: float = 1e-22,
    training_ids: set[str] | None = None,
    held_out_ids: set[str] | None = None,
    numerical_uncertainty: float = NUMERICAL_SENSOR_UNCERTAINTY,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if training_ids is None:
        training_ids = {"aortic_root", "thoracic_aorta", "femoral", "iliac"}
    if held_out_ids is None:
        held_out_ids = {"carotid", "brachial"}
    if training_ids & held_out_ids:
        raise ValueError("training and held-out sets overlap")

    targets: dict[str, np.ndarray] = {}
    work_targets: dict[str, np.ndarray] = {}
    shear_signals: dict[str, np.ndarray] = {}
    for case in cases:
        hydro = hydrodynamics[case.artery_id]
        force = np.asarray(hydro["force_signed_n"])
        shear = np.asarray(hydro["wall_shear_pa"])
        psi_target = wss_work(
            shear, wss_activation_volume_m3, sensor.temperature_k
        ) + lamb_work(force, coupling_length_m, sensor.temperature_k)
        target, _ = _sensor_from_work(psi_target, fluid, sensor)
        targets[case.artery_id] = target
        work_targets[case.artery_id] = psi_target
        shear_signals[case.artery_id] = shear

    n = len(next(iter(shear_signals.values())))
    best: dict[str, float] | None = None
    for lag in range(-n // 2, n // 2):
        z_parts = [np.roll(shear_signals[cid], lag) for cid in sorted(training_ids)]
        y_parts = [work_targets[cid] for cid in sorted(training_ids)]
        z = np.concatenate(z_parts)
        y = np.concatenate(y_parts)
        z_center = z - z.mean()
        denominator = float(np.dot(z_center, z_center))
        gain = float(np.dot(z_center, y - y.mean()) / max(denominator, 1e-30))
        offset = float(y.mean() - gain * z.mean())
        mse = float(np.mean((gain * z + offset - y) ** 2))
        if best is None or mse < best["training_work_mse"]:
            best = {
                "lag_samples": float(lag),
                "lag_cycle": float(lag / n),
                "gain_work_per_pa": gain,
                "offset_work": offset,
                "training_work_mse": mse,
            }
    assert best is not None

    rows: list[dict[str, Any]] = []
    for case in cases:
        cid = case.artery_id
        psi_pred = (
            best["gain_work_per_pa"]
            * np.roll(shear_signals[cid], int(best["lag_samples"]))
            + best["offset_work"]
        )
        prediction, _ = _sensor_from_work(psi_pred, fluid, sensor)
        target = targets[cid]
        residual = rms_difference(prediction, target)
        target_range = max(signal_metrics(target)["dynamic_range"], 1e-30)
        rows.append(
            {
                "artery_id": cid,
                "artery_name": case.name,
                "split": "train" if cid in training_ids else "held_out",
                "rms_residual": residual,
                "fraction_target_dynamic_range": residual / target_range,
                "passes_E2": bool(
                    residual >= 0.005
                    and residual / target_range >= 0.05
                    and residual >= 10.0 * numerical_uncertainty
                ),
            }
        )
    return pd.DataFrame(rows), best
