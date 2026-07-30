from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES
from scipy.stats import spearmanr

from .core import (
    AnalysisConfig,
    alpha_for_case,
    canonical_coefficients,
    combine_unordered_pairs,
    critical_anisotropy,
    eta_for_case,
    evaluate_kernel,
    exact_excess_kernel,
    force_scale,
    near_wall_basis,
    relative_l2,
    rms,
    second_order_kernel,
    susceptibility_from_kernel,
    unit_perturbation_response,
)

_EPS = 1e-30


def _case_by_id(identifier: str) -> Any:
    for case in V2_ARTERY_CASES:
        if case.artery_id == identifier:
            return case
    raise KeyError(identifier)


def input_rms(coefficients: Sequence[complex]) -> float:
    coefficients = np.asarray(coefficients, dtype=complex)
    return float(np.sqrt(0.5 * np.sum(np.abs(coefficients) ** 2)))


def normalise_input_rms(
    coefficients: Sequence[complex], target_rms: float
) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=complex)
    current = input_rms(coefficients)
    if current <= 0.0:
        raise ValueError("cannot normalise a zero waveform")
    return coefficients * (target_rms / current)


def waveform_catalogue(random_seed: int = 20260730) -> list[dict[str, Any]]:
    catalogue: list[dict[str, Any]] = []
    for case in V2_ARTERY_CASES:
        catalogue.append(
            {
                "waveform_id": f"native_{case.artery_id}",
                "family": "native",
                "source_artery": case.artery_id,
                "coefficients": np.asarray(case.harmonic_coefficients, dtype=complex),
            }
        )

    target = 1.0
    for harmonic in range(1, 7):
        coefficients = np.zeros(6, dtype=complex)
        coefficients[harmonic - 1] = np.sqrt(2.0) * target
        catalogue.append(
            {
                "waveform_id": f"single_h{harmonic}",
                "family": "single_tone",
                "source_artery": None,
                "coefficients": coefficients,
            }
        )

    for first in range(1, 7):
        for second in range(first + 1, 7):
            coefficients = np.zeros(6, dtype=complex)
            coefficients[[first - 1, second - 1]] = 1.0
            coefficients = normalise_input_rms(coefficients, target)
            catalogue.append(
                {
                    "waveform_id": f"two_h{first}_h{second}",
                    "family": "two_tone",
                    "source_artery": None,
                    "coefficients": coefficients,
                }
            )

    for harmonics in ((1, 2, 3), (1, 3, 5), (2, 4, 6)):
        coefficients = np.zeros(6, dtype=complex)
        coefficients[np.asarray(harmonics) - 1] = 1.0
        coefficients = normalise_input_rms(coefficients, target)
        catalogue.append(
            {
                "waveform_id": "three_" + "_".join(map(str, harmonics)),
                "family": "sparse_three_tone",
                "source_artery": None,
                "coefficients": coefficients,
            }
        )

    harmonic_numbers = np.arange(1, 7, dtype=float)
    for exponent in np.linspace(0.0, 2.0, 5):
        coefficients = harmonic_numbers ** (-exponent)
        coefficients = normalise_input_rms(coefficients, target)
        catalogue.append(
            {
                "waveform_id": f"slope_{exponent:.2f}",
                "family": "spectral_slope",
                "source_artery": None,
                "coefficients": coefficients.astype(complex),
            }
        )

    generator = np.random.default_rng(random_seed)
    for case in V2_ARTERY_CASES:
        amplitudes = np.abs(np.asarray(case.harmonic_coefficients, dtype=complex))
        for index in range(9):
            phases = generator.uniform(-np.pi, np.pi, size=6)
            catalogue.append(
                {
                    "waveform_id": f"phase_{case.artery_id}_{index + 1:02d}",
                    "family": "phase_challenge",
                    "source_artery": case.artery_id,
                    "coefficients": amplitudes * np.exp(1j * phases),
                }
            )
    return catalogue


def build_operator_samples(
    config: AnalysisConfig = AnalysisConfig(),
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    config.validate()
    native_eta = np.asarray([eta_for_case(case) for case in V2_ARTERY_CASES])
    reference_eta = float(np.median(native_eta))
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for case in V2_ARTERY_CASES:
        unit = unit_perturbation_response(case, config)
        for condition, eta in (
            ("reference", reference_eta),
            ("physiological", eta_for_case(case)),
        ):
            near_wall = near_wall_basis(case, unit, config, eta)
            frequencies, kernel = second_order_kernel(case, near_wall)
            records.append(
                {
                    "condition": condition,
                    "artery_id": case.artery_id,
                    "artery_name": case.name,
                    "alpha": alpha_for_case(case),
                    "eta": eta,
                    "frequencies": frequencies,
                    "kernel": kernel,
                    "kernel_norm": float(np.linalg.norm(kernel)),
                    "maximum_residual": unit.maximum_residual,
                }
            )
            arrays[f"{condition}__{case.artery_id}__frequencies"] = frequencies
            arrays[f"{condition}__{case.artery_id}__kernel"] = kernel
    return records, arrays


def native_atlas(
    records: Sequence[dict[str, Any]],
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["condition"] != "physiological":
            continue
        case = _case_by_id(record["artery_id"])
        result = susceptibility_from_kernel(
            case,
            record["frequencies"],
            record["kernel"],
            case.harmonic_coefficients,
            config,
        )
        coefficient_rms_n = rms(result.waveform)
        coefficient_peak_n = float(np.max(np.abs(result.waveform)))
        rows.append(
            {
                "artery_id": case.artery_id,
                "artery_name": case.name,
                "alpha": record["alpha"],
                "eta": record["eta"],
                "phi_rms": result.rms,
                "phi_peak_absolute": result.peak_absolute,
                "outward_duty": result.outward_duty,
                "inward_duty": result.inward_duty,
                "force_coefficient_rms_n_per_epsilon2": coefficient_rms_n,
                "force_coefficient_peak_n_per_epsilon2": coefficient_peak_n,
                "predicted_rms_at_epsilon_0p08_pn": (
                    coefficient_rms_n * 0.08**2 * 1e12
                ),
                "critical_epsilon_1pn_rms": critical_anisotropy(
                    1e-12, coefficient_rms_n
                ),
                "critical_epsilon_10pn_rms": critical_anisotropy(
                    10e-12, coefficient_rms_n
                ),
            }
        )
    return pd.DataFrame(rows)


def crossed_matrices(
    records: Sequence[dict[str, Any]],
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        vessel = _case_by_id(record["artery_id"])
        for waveform_case in V2_ARTERY_CASES:
            result = susceptibility_from_kernel(
                vessel,
                record["frequencies"],
                record["kernel"],
                waveform_case.harmonic_coefficients,
                config,
            )
            rows.append(
                {
                    "condition": record["condition"],
                    "vessel_id": vessel.artery_id,
                    "vessel_name": vessel.name,
                    "waveform_id": waveform_case.artery_id,
                    "waveform_name": waveform_case.name,
                    "native_diagonal": vessel.artery_id == waveform_case.artery_id,
                    "alpha": record["alpha"],
                    "eta": record["eta"],
                    "phi_rms": result.rms,
                    "phi_peak_absolute": result.peak_absolute,
                }
            )
    return pd.DataFrame(rows)


def waveform_controls(
    records: Sequence[dict[str, Any]],
    config: AnalysisConfig = AnalysisConfig(),
    random_seed: int = 20260730,
) -> pd.DataFrame:
    generator = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["condition"] != "physiological":
            continue
        case = _case_by_id(record["artery_id"])
        native = np.asarray(case.harmonic_coefficients, dtype=complex)
        native_result = susceptibility_from_kernel(
            case, record["frequencies"], record["kernel"], native, config
        )
        target_rms = input_rms(native)
        controls: list[tuple[str, str, np.ndarray]] = [
            ("native", "native", native),
            ("sign_neutralised", "sign", np.abs(native).astype(complex)),
            (
                "common_phase_pi_over_4",
                "phase",
                np.abs(native) * np.exp(1j * np.pi / 4.0),
            ),
        ]
        for harmonic in range(6):
            removed = native.copy()
            removed[harmonic] = 0.0
            controls.append(
                (f"remove_h{harmonic + 1}", "harmonic_removal", removed)
            )
            controls.append(
                (
                    f"remove_h{harmonic + 1}_rms_matched",
                    "harmonic_removal_rms_matched",
                    normalise_input_rms(removed, target_rms),
                )
            )
        for index in range(8):
            phases = generator.uniform(-np.pi, np.pi, size=6)
            controls.append(
                (
                    f"phase_random_{index + 1:02d}",
                    "phase",
                    np.abs(native) * np.exp(1j * phases),
                )
            )

        for name, family, coefficients in controls:
            result = susceptibility_from_kernel(
                case,
                record["frequencies"],
                record["kernel"],
                coefficients,
                config,
            )
            rows.append(
                {
                    "artery_id": case.artery_id,
                    "artery_name": case.name,
                    "control": name,
                    "family": family,
                    "input_rms": input_rms(coefficients),
                    "phi_rms": result.rms,
                    "relative_to_native": result.rms / native_result.rms,
                    "fractional_change": result.rms / native_result.rms - 1.0,
                }
            )
    return pd.DataFrame(rows)


def harmonic_pair_attribution(
    records: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["condition"] != "physiological":
            continue
        case = _case_by_id(record["artery_id"])
        frequencies, coefficients = canonical_coefficients(case.harmonic_coefficients)
        output_frequencies, spectrum, ordered = evaluate_kernel(
            frequencies, record["kernel"], coefficients
        )
        output_lookup = {int(q): value for q, value in zip(output_frequencies, spectrum)}
        pairs = combine_unordered_pairs(frequencies, ordered)
        by_output: dict[int, list[dict[str, Any]]] = {}
        for pair in pairs:
            by_output.setdefault(int(pair["q"]), []).append(pair)
        for output, candidates in by_output.items():
            if output < 0:
                continue
            total_absolute = sum(abs(complex(item["contribution"])) for item in candidates)
            ranked = sorted(
                candidates,
                key=lambda item: abs(complex(item["contribution"])),
                reverse=True,
            )
            for rank, item in enumerate(ranked[:5], start=1):
                contribution = complex(item["contribution"])
                rows.append(
                    {
                        "artery_id": case.artery_id,
                        "artery_name": case.name,
                        "output_frequency": output,
                        "rank": rank,
                        "m": item["m"],
                        "n": item["n"],
                        "contribution_real_n": contribution.real,
                        "contribution_imag_n": contribution.imag,
                        "contribution_absolute_n": abs(contribution),
                        "fraction_of_absolute_pair_sum": abs(contribution)
                        / max(total_absolute, _EPS),
                        "output_absolute_n": abs(output_lookup.get(output, 0.0)),
                    }
                )
    return pd.DataFrame(rows)


def _fit_scale(records: Sequence[dict[str, Any]]) -> np.ndarray:
    design = np.asarray(
        [
            [1.0, np.log(record["alpha"]), np.log(record["eta"])]
            for record in records
        ]
    )
    response = np.log([record["kernel_norm"] for record in records])
    return np.linalg.lstsq(design, response, rcond=None)[0]


def _predict_scale(parameters: Sequence[float], alpha: float, eta: float) -> float:
    log_prefactor, alpha_exponent, eta_exponent = np.asarray(parameters, dtype=float)
    return float(np.exp(log_prefactor) * alpha**alpha_exponent * eta**eta_exponent)


def _normalised_mean_kernel(records: Iterable[dict[str, Any]]) -> np.ndarray:
    kernels = [
        record["kernel"] / max(record["kernel_norm"], _EPS) for record in records
    ]
    if not kernels:
        raise ValueError("at least one operator sample is required")
    return np.mean(np.stack(kernels), axis=0)


def _rank_one(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    left, singular_values, right = np.linalg.svd(kernel, full_matrices=False)
    reduced = singular_values[0] * np.outer(left[:, 0], right[0, :])
    retained = float(singular_values[0] ** 2 / np.sum(singular_values**2))
    return reduced, singular_values, retained


def reduced_law_validation(
    records: Sequence[dict[str, Any]],
    config: AnalysisConfig = AnalysisConfig(),
    catalogue: Sequence[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    catalogue = waveform_catalogue() if catalogue is None else list(catalogue)
    predictions: list[dict[str, Any]] = []
    exponent_rows: list[dict[str, Any]] = []
    artery_ids = [case.artery_id for case in V2_ARTERY_CASES]
    for held_out in artery_ids:
        training = [record for record in records if record["artery_id"] != held_out]
        testing = [record for record in records if record["artery_id"] == held_out]
        parameters = _fit_scale(training)
        universal = _normalised_mean_kernel(training)
        reduced, _singular_values, retained = _rank_one(universal)
        exponent_rows.append(
            {
                "held_out_artery": held_out,
                "prefactor": float(np.exp(parameters[0])),
                "alpha_exponent": float(parameters[1]),
                "eta_exponent": float(parameters[2]),
                "retained_energy": retained,
            }
        )
        for record in testing:
            scale = _predict_scale(parameters, record["alpha"], record["eta"])
            case = _case_by_id(record["artery_id"])
            for waveform in catalogue:
                exact = susceptibility_from_kernel(
                    case,
                    record["frequencies"],
                    record["kernel"],
                    waveform["coefficients"],
                    config,
                ).rms
                predicted = susceptibility_from_kernel(
                    case,
                    record["frequencies"],
                    scale * reduced,
                    waveform["coefficients"],
                    config,
                ).rms
                predictions.append(
                    {
                        "held_out_artery": held_out,
                        "condition": record["condition"],
                        "waveform_id": waveform["waveform_id"],
                        "family": waveform["family"],
                        "exact_phi_rms": exact,
                        "predicted_phi_rms": predicted,
                        "relative_error": abs(predicted - exact) / max(exact, _EPS),
                        "retained_energy": retained,
                    }
                )

    full_parameters = _fit_scale(records)
    full_universal = _normalised_mean_kernel(records)
    full_reduced, singular_values, retained = _rank_one(full_universal)
    prediction_frame = pd.DataFrame(predictions)
    native = prediction_frame[prediction_frame["family"] == "native"]
    correlations = []
    for held_out, group in native.groupby("held_out_artery"):
        correlations.append(
            float(
                spearmanr(group["exact_phi_rms"], group["predicted_phi_rms"]).statistic
            )
        )
    law = {
        "prefactor": float(np.exp(full_parameters[0])),
        "alpha_exponent": float(full_parameters[1]),
        "eta_exponent": float(full_parameters[2]),
        "retained_energy": retained,
        "median_relative_error": float(prediction_frame["relative_error"].median()),
        "p90_relative_error": float(prediction_frame["relative_error"].quantile(0.90)),
        "maximum_relative_error": float(prediction_frame["relative_error"].max()),
        "minimum_native_spearman": float(min(correlations)),
        "leave_one_out_exponents": exponent_rows,
    }
    arrays = {
        "universal_kernel": full_universal,
        "rank_one_kernel": full_reduced,
        "singular_values": singular_values,
        "scale_parameters": full_parameters,
    }
    return prediction_frame, law, arrays


def constitutive_robustness(
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    epsilon = config.validation_epsilon
    paths = (
        ("reciprocal", 1.0, 1.0, 1.0),
        ("beta_low", 0.5, 1.0, 1.0),
        ("gamma_low", 1.0, 0.5, 1.0),
        ("gamma_only", 0.0, 1.0, 1.0),
        ("beta_only", 1.0, 0.0, 1.0),
        ("beta_high_gamma_low", 1.25, 0.75, 1.0),
        ("beta_low_gamma_high", 0.75, 1.25, 1.0),
        ("delta_low", 1.0, 1.0, 0.8),
        ("delta_high", 1.0, 1.0, 1.2),
    )
    rows: list[dict[str, Any]] = []
    for case in V2_ARTERY_CASES:
        perturbation = near_wall_basis(
            case, unit_perturbation_response(case, config), config
        )
        frequencies, reciprocal_second_order = second_order_kernel(case, perturbation)
        reciprocal_norm = np.linalg.norm(reciprocal_second_order)
        reciprocal_shape = reciprocal_second_order / max(reciprocal_norm, _EPS)
        for name, beta_factor, gamma_factor, delta in paths:
            exact_frequencies, exact_kernel, residual = exact_excess_kernel(
                case,
                beta_factor * epsilon,
                gamma_factor * epsilon,
                delta,
                config,
            )
            if not np.array_equal(frequencies, exact_frequencies):
                raise RuntimeError("frequency axes disagree")
            scaled = exact_kernel / epsilon**2
            scaled_norm = np.linalg.norm(scaled)
            shape = scaled / max(scaled_norm, _EPS)
            null_control = gamma_factor == 0.0
            rows.append(
                {
                    "artery_id": case.artery_id,
                    "artery_name": case.name,
                    "constitutive_path": name,
                    "beta_factor": beta_factor,
                    "gamma_factor": gamma_factor,
                    "delta": delta,
                    "null_control": null_control,
                    "scaled_kernel_norm": float(scaled_norm),
                    "relative_amplitude_to_reciprocal": float(
                        scaled_norm / max(reciprocal_norm, _EPS)
                    ),
                    "normalised_shape_relative_l2": 0.0
                    if null_control and scaled_norm <= 1e-20
                    else relative_l2(shape, reciprocal_shape),
                    "maximum_residual": residual,
                }
            )
    return pd.DataFrame(rows)


def run_analysis(
    output_root: str | Path,
    config: AnalysisConfig = AnalysisConfig(),
) -> dict[str, Any]:
    """Execute the complete public analysis and write reusable result files."""

    config.validate()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records, operator_arrays = build_operator_samples(config)
    atlas = native_atlas(records, config)
    crossed = crossed_matrices(records, config)
    controls = waveform_controls(records, config)
    pairs = harmonic_pair_attribution(records)
    predictions, law, reduction_arrays = reduced_law_validation(records, config)
    robustness = constitutive_robustness(config)

    tables = {
        "artery_atlas.csv": atlas,
        "crossed_susceptibility.csv": crossed,
        "waveform_controls.csv": controls,
        "harmonic_pair_attribution.csv": pairs,
        "reduced_law_validation.csv": predictions,
        "constitutive_robustness.csv": robustness,
    }
    for name, frame in tables.items():
        frame.to_csv(output_root / name, index=False)

    np.savez_compressed(
        output_root / "operator_archive.npz", **operator_arrays, **reduction_arrays
    )
    summary = {
        "software": "piconewton-waveform-susceptibility",
        "configuration": asdict(config),
        "arteries": int(atlas["artery_id"].nunique()),
        "crossed_entries": int(len(crossed)),
        "reduced_law": law,
        "scientific_scope": (
            "straight rigid axisymmetric six-harmonic anisotropic Womersley model"
        ),
        "claim_boundary": (
            "the reciprocal amplitude prefactor applies to beta=gamma and delta=1; "
            "other tensors require a separate constitutive amplitude factor"
        ),
    }
    (output_root / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "output_root": str(output_root),
        "atlas": atlas,
        "crossed": crossed,
        "controls": controls,
        "pairs": pairs,
        "predictions": predictions,
        "robustness": robustness,
        "summary": summary,
    }
