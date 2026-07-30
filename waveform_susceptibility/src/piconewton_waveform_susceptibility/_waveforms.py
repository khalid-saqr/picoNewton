from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES

from .core import (
    AnalysisConfig,
    canonical_coefficients,
    combine_unordered_pairs,
    critical_anisotropy,
    evaluate_kernel,
    rms,
    susceptibility_from_kernel,
)

_EPS = 1e-30


def _case_by_id(identifier: str) -> Any:
    for case in V2_ARTERY_CASES:
        if case.artery_id == identifier:
            return case
    raise KeyError(identifier)


def input_rms(coefficients: Sequence[complex]) -> float:
    values = np.asarray(coefficients, dtype=complex)
    return float(np.sqrt(0.5 * np.sum(np.abs(values) ** 2)))


def normalise_input_rms(
    coefficients: Sequence[complex], target_rms: float
) -> np.ndarray:
    values = np.asarray(coefficients, dtype=complex)
    current = input_rms(values)
    if current <= 0.0:
        raise ValueError("cannot normalise a zero waveform")
    return values * (target_rms / current)


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

    target_rms = 1.0
    for harmonic in range(1, 7):
        coefficients = np.zeros(6, dtype=complex)
        coefficients[harmonic - 1] = np.sqrt(2.0) * target_rms
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
            catalogue.append(
                {
                    "waveform_id": f"two_h{first}_h{second}",
                    "family": "two_tone",
                    "source_artery": None,
                    "coefficients": normalise_input_rms(coefficients, target_rms),
                }
            )

    for harmonics in ((1, 2, 3), (1, 3, 5), (2, 4, 6)):
        coefficients = np.zeros(6, dtype=complex)
        coefficients[np.asarray(harmonics) - 1] = 1.0
        catalogue.append(
            {
                "waveform_id": "three_" + "_".join(map(str, harmonics)),
                "family": "sparse_three_tone",
                "source_artery": None,
                "coefficients": normalise_input_rms(coefficients, target_rms),
            }
        )

    harmonic_numbers = np.arange(1, 7, dtype=float)
    for exponent in np.linspace(0.0, 2.0, 5):
        coefficients = normalise_input_rms(
            harmonic_numbers ** (-exponent), target_rms
        )
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
        coefficient_rms = rms(result.waveform)
        coefficient_peak = float(np.max(np.abs(result.waveform)))
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
                "force_coefficient_rms_n_per_epsilon2": coefficient_rms,
                "force_coefficient_peak_n_per_epsilon2": coefficient_peak,
                "predicted_rms_at_epsilon_0p08_pn": coefficient_rms * 0.08**2 * 1e12,
                "critical_epsilon_1pn_rms": critical_anisotropy(
                    1e-12, coefficient_rms
                ),
                "critical_epsilon_10pn_rms": critical_anisotropy(
                    10e-12, coefficient_rms
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
            controls.append((f"remove_h{harmonic + 1}", "harmonic_removal", removed))
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


def harmonic_pair_attribution(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["condition"] != "physiological":
            continue
        case = _case_by_id(record["artery_id"])
        frequencies, coefficients = canonical_coefficients(case.harmonic_coefficients)
        output_frequencies, spectrum, ordered = evaluate_kernel(
            frequencies, record["kernel"], coefficients
        )
        output_lookup = {
            int(q): value
            for q, value in zip(output_frequencies, spectrum, strict=True)
        }
        by_output: dict[int, list[dict[str, Any]]] = {}
        for pair in combine_unordered_pairs(frequencies, ordered):
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
