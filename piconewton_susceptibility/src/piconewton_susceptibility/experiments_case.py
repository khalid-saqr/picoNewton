from __future__ import annotations

from typing import Any

import numpy as np
from piconewton_v3 import V2_ARTERY_CASES

from .experiments_core import (
    Step7Config,
    causal_waveform_families,
    evaluate_susceptibility,
    exact_second_order_error,
    input_rms,
    normalise_input_rms,
    susceptibility_metrics,
)


def crossed_rows(
    vessel: Any,
    matrix_type: str,
    eta: float,
    frequencies: np.ndarray,
    kernel2: np.ndarray,
    kernel_exact: np.ndarray,
    config: Step7Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    matrix_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for waveform_case in V2_ARTERY_CASES:
        q, spectrum2, waveform2 = evaluate_susceptibility(
            frequencies, kernel2, waveform_case.harmonic_coefficients, config.time_points
        )
        q_exact, spectrum_exact, waveform_exact = evaluate_susceptibility(
            frequencies, kernel_exact, waveform_case.harmonic_coefficients, config.time_points
        )
        if not np.array_equal(q, q_exact):
            raise RuntimeError("output frequency axes disagree")
        metrics = susceptibility_metrics(waveform2, spectrum2)
        matrix_rows.append(
            {
                "matrix_type": matrix_type,
                "vessel_id": vessel.artery_id,
                "vessel_name": vessel.name,
                "waveform_id": waveform_case.artery_id,
                "waveform_name": waveform_case.name,
                "native_diagonal": vessel.artery_id == waveform_case.artery_id,
                "eta": eta,
                **metrics,
            }
        )
        exact_rows.append(
            {
                "matrix_type": matrix_type,
                "vessel_id": vessel.artery_id,
                "waveform_id": waveform_case.artery_id,
                "epsilon": config.exact_epsilon,
                **exact_second_order_error(waveform_exact, waveform2, config.exact_epsilon),
                "spectrum_relative_l2": float(
                    np.linalg.norm(spectrum_exact / config.exact_epsilon**2 - spectrum2)
                    / max(np.linalg.norm(spectrum2), 1e-30)
                ),
            }
        )
        arrays[
            f"{matrix_type}__{vessel.artery_id}__{waveform_case.artery_id}__phi"
        ] = waveform2
    return matrix_rows, exact_rows, arrays


def native_control_rows(
    vessel: Any,
    frequencies: np.ndarray,
    kernel2: np.ndarray,
    config: Step7Config,
    phase_scrambles: list[np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    native = np.asarray(vessel.harmonic_coefficients, dtype=complex)
    _, native_spectrum, native_waveform = evaluate_susceptibility(
        frequencies, kernel2, native, config.time_points
    )
    native_rms = susceptibility_metrics(native_waveform, native_spectrum)["phi_rms"]
    target_input_rms = input_rms(native)
    controls: list[tuple[str, str, np.ndarray]] = [
        ("native", "native", native),
        ("sign_neutralized", "sign", np.abs(native)),
        ("phase_aligned_zero", "phase", np.abs(native).astype(complex)),
        ("phase_coherent_common_pi4", "phase", np.abs(native) * np.exp(1j * np.pi / 4.0)),
    ]
    for index in range(6):
        removed = native.copy()
        removed[index] = 0.0
        controls.append((f"remove_h{index + 1}", "harmonic_removal", removed))
        controls.append(
            (
                f"remove_h{index + 1}_rms_matched",
                "harmonic_removal_rms_matched",
                normalise_input_rms(removed, target_input_rms),
            )
        )
    for index, phases in enumerate(phase_scrambles, start=1):
        controls.append(
            (
                f"phase_scramble_{index:02d}",
                "phase",
                np.abs(native) * np.exp(1j * phases),
            )
        )
    rows: list[dict[str, Any]] = []
    by_name: dict[str, float] = {}
    for control_name, family, coefficients in controls:
        _q, spectrum, waveform = evaluate_susceptibility(
            frequencies, kernel2, coefficients, config.time_points
        )
        metrics = susceptibility_metrics(waveform, spectrum)
        by_name[control_name] = metrics["phi_rms"]
        rows.append(
            {
                "vessel_id": vessel.artery_id,
                "waveform_source": vessel.artery_id,
                "control": control_name,
                "family": family,
                "input_rms": input_rms(coefficients),
                "relative_to_native_rms": metrics["phi_rms"] / native_rms,
                "fractional_change_from_native": (metrics["phi_rms"] - native_rms)
                / native_rms,
                **metrics,
            }
        )
    degeneracy = {
        "vessel_id": vessel.artery_id,
        "sign_neutralized_equals_zero_phase_relative_error": abs(
            by_name["sign_neutralized"] - by_name["phase_aligned_zero"]
        )
        / max(by_name["sign_neutralized"], 1e-30),
        "interpretation": "expected algebraic identity for real signed native coefficients",
    }
    return rows, degeneracy


def causal_family_rows(
    vessel: Any,
    frequencies: np.ndarray,
    kernel2: np.ndarray,
    config: Step7Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for control_name, family, coefficients in causal_waveform_families():
        _q, spectrum, waveform = evaluate_susceptibility(
            frequencies, kernel2, coefficients, config.time_points
        )
        rows.append(
            {
                "vessel_id": vessel.artery_id,
                "control": control_name,
                "family": family,
                "input_rms": input_rms(coefficients),
                **susceptibility_metrics(waveform, spectrum),
            }
        )
    return rows
