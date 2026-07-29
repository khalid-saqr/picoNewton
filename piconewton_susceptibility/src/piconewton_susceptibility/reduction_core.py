from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from piconewton_v3 import V2_ARTERY_CASES
from scipy.optimize import minimize_scalar

_EPS = 1e-30
FREQUENCIES = np.concatenate((np.arange(-6, 0), np.arange(1, 7))).astype(int)
OUTPUT_FREQUENCIES = np.arange(-12, 13, dtype=int)


@dataclass(frozen=True)
class Step8Config:
    profile: str = "publication"
    random_seed: int = 20260730
    phase_scrambles: int = 8
    candidate_ranks: tuple[int, ...] = (1, 2, 3)
    retained_energy_min: float = 0.999
    median_relative_error_max: float = 0.05
    p90_relative_error_max: float = 0.12
    maximum_relative_error_max: float = 0.20
    family_median_relative_error_max: float = 0.05
    family_maximum_relative_error_max: float = 0.20
    ranking_spearman_min: float = 0.95
    exponent_span_max: float = 0.08

    def validate(self) -> None:
        if self.profile not in {"quick", "publication"}:
            raise ValueError("profile must be quick or publication")
        if self.phase_scrambles < 4:
            raise ValueError("at least four phase scrambles are required")
        if (
            not self.candidate_ranks
            or min(self.candidate_ranks) < 1
            or max(self.candidate_ranks) > 3
        ):
            raise ValueError("candidate ranks must be within 1..3")
        for value in (
            self.retained_energy_min,
            self.median_relative_error_max,
            self.p90_relative_error_max,
            self.maximum_relative_error_max,
            self.family_median_relative_error_max,
            self.family_maximum_relative_error_max,
            self.ranking_spearman_min,
            self.exponent_span_max,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("all thresholds must be positive and finite")


def canonical_coefficients(one_sided: Sequence[complex]) -> np.ndarray:
    values = np.asarray(one_sided, dtype=complex)
    if values.shape != (6,):
        raise ValueError("six one-sided harmonic coefficients are required")
    result = np.empty(12, dtype=complex)
    result[:6] = np.conj(values[::-1]) / 2.0
    result[6:] = values / 2.0
    return result


def input_rms(one_sided: Sequence[complex]) -> float:
    values = np.asarray(one_sided, dtype=complex)
    return float(np.sqrt(0.5 * np.sum(np.abs(values) ** 2)))


def normalise_input_rms(one_sided: Sequence[complex], target: float = 1.0) -> np.ndarray:
    values = np.asarray(one_sided, dtype=complex)
    return values * target / max(input_rms(values), _EPS)


def waveform_catalog(config: Step8Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in V2_ARTERY_CASES:
        rows.append(
            {
                "waveform_id": f"native_{case.artery_id}",
                "family": "native",
                "source_artery": case.artery_id,
                "coefficients": np.asarray(case.harmonic_coefficients, dtype=complex),
            }
        )
    for harmonic in range(1, 7):
        coefficients = np.zeros(6, dtype=complex)
        coefficients[harmonic - 1] = 1.0
        rows.append(
            {
                "waveform_id": f"single_h{harmonic}",
                "family": "single_tone",
                "source_artery": None,
                "coefficients": normalise_input_rms(coefficients),
            }
        )
    for first in range(1, 7):
        for second in range(first + 1, 7):
            coefficients = np.zeros(6, dtype=complex)
            coefficients[first - 1] = 1.0
            coefficients[second - 1] = np.exp(1j * np.pi / 3.0)
            rows.append(
                {
                    "waveform_id": f"two_h{first}_h{second}",
                    "family": "two_tone",
                    "source_artery": None,
                    "coefficients": normalise_input_rms(coefficients),
                }
            )
    for name, harmonics in (
        ("three_123", (1, 2, 3)),
        ("three_135", (1, 3, 5)),
        ("three_246", (2, 4, 6)),
    ):
        coefficients = np.zeros(6, dtype=complex)
        for index, harmonic in enumerate(harmonics):
            coefficients[harmonic - 1] = np.exp(1j * index * np.pi / 4.0)
        rows.append(
            {
                "waveform_id": name,
                "family": "sparse_three_tone",
                "source_artery": None,
                "coefficients": normalise_input_rms(coefficients),
            }
        )
    for slope in (0.0, 0.5, 1.0, 1.5, 2.0):
        coefficients = np.arange(1, 7, dtype=float) ** (-slope)
        rows.append(
            {
                "waveform_id": f"slope_{slope:.1f}",
                "family": "spectral_slope",
                "source_artery": None,
                "coefficients": normalise_input_rms(coefficients),
            }
        )
    rng = np.random.default_rng(config.random_seed)
    phase_vectors = [rng.uniform(-np.pi, np.pi, 6) for _ in range(config.phase_scrambles)]
    for case in V2_ARTERY_CASES:
        magnitude = np.abs(np.asarray(case.harmonic_coefficients, dtype=complex))
        rows.append(
            {
                "waveform_id": f"phase_{case.artery_id}_common_pi4",
                "family": "phase_challenge",
                "source_artery": case.artery_id,
                "coefficients": magnitude * np.exp(1j * np.pi / 4.0),
            }
        )
        for index, phases in enumerate(phase_vectors, start=1):
            rows.append(
                {
                    "waveform_id": f"phase_{case.artery_id}_scramble_{index:02d}",
                    "family": "phase_challenge",
                    "source_artery": case.artery_id,
                    "coefficients": magnitude * np.exp(1j * phases),
                }
            )
    return rows


def spectrum_from_kernel(kernel: np.ndarray, one_sided: Sequence[complex]) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=complex)
    if kernel.shape != (12, 12):
        raise ValueError("the interaction kernel must be 12 by 12")
    coefficients = canonical_coefficients(one_sided)
    spectrum = np.zeros(25, dtype=complex)
    for i, first in enumerate(FREQUENCIES):
        for j, second in enumerate(FREQUENCIES):
            spectrum[int(first + second) + 12] += (
                kernel[i, j] * coefficients[i] * coefficients[j]
            )
    return spectrum


def susceptibility_from_kernel(kernel: np.ndarray, one_sided: Sequence[complex]) -> float:
    spectrum = spectrum_from_kernel(kernel, one_sided)
    return float(np.sqrt(np.sum(np.abs(spectrum) ** 2)))


def kernel_scale(kernel: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(kernel), ord="fro"))


def fit_power_law(alpha: np.ndarray, eta: np.ndarray, scale: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(alpha)), np.log(alpha), np.log(eta)))
    return np.linalg.lstsq(design, np.log(scale), rcond=None)[0]


def predict_power_law(parameters: Sequence[float], alpha: float, eta: float) -> float:
    log_c, p_alpha, p_eta = np.asarray(parameters, dtype=float)
    return float(np.exp(log_c) * alpha**p_alpha * eta**p_eta)


def universal_kernel(kernels: Iterable[np.ndarray]) -> np.ndarray:
    normalised = []
    for kernel in kernels:
        scale = kernel_scale(kernel)
        normalised.append(np.asarray(kernel, dtype=complex) / max(scale, _EPS))
    if not normalised:
        raise ValueError("at least one kernel is required")
    return np.mean(np.stack(normalised), axis=0)


def truncated_kernel(kernel: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, float]:
    if rank < 1 or rank > 3:
        raise ValueError("rank must lie in 1..3")
    u, singular_values, vh = np.linalg.svd(np.asarray(kernel, dtype=complex), full_matrices=False)
    reduced = (u[:, :rank] * singular_values[:rank]) @ vh[:rank, :]
    retained = float(
        np.sum(singular_values[:rank] ** 2) / max(np.sum(singular_values**2), _EPS)
    )
    return reduced, singular_values, retained


def inverse_harmonic_moment(one_sided: Sequence[complex], exponent: float) -> float:
    coefficients = np.asarray(one_sided, dtype=complex)
    harmonics = np.arange(1, 7, dtype=float)
    return float(np.sum(np.abs(coefficients) ** 2 * harmonics ** (-exponent)))


def fit_scalar_moment(
    rows: Sequence[tuple[float, float, np.ndarray, float]],
) -> tuple[np.ndarray, float]:
    if not rows:
        raise ValueError("training rows are empty")

    def fit_at(exponent: float) -> tuple[float, np.ndarray]:
        design = np.array(
            [[1.0, np.log(alpha), np.log(eta)] for alpha, eta, _g, _phi in rows]
        )
        adjusted = np.array(
            [
                np.log(phi) - np.log(max(inverse_harmonic_moment(g, exponent), _EPS))
                for alpha, eta, g, phi in rows
            ]
        )
        parameters = np.linalg.lstsq(design, adjusted, rcond=None)[0]
        fitted = design @ parameters + np.array(
            [np.log(max(inverse_harmonic_moment(g, exponent), _EPS)) for _a, _e, g, _p in rows]
        )
        truth = np.log([phi for _a, _e, _g, phi in rows])
        return float(np.mean((fitted - truth) ** 2)), parameters

    result = minimize_scalar(
        lambda exponent: fit_at(exponent)[0],
        bounds=(-1.0, 5.0),
        method="bounded",
        options={"xatol": 1e-7},
    )
    _loss, parameters = fit_at(float(result.x))
    return parameters, float(result.x)


def predict_scalar_moment(
    parameters: Sequence[float], exponent: float, alpha: float, eta: float, coefficients: np.ndarray
) -> float:
    scale = predict_power_law(parameters, alpha, eta)
    return scale * inverse_harmonic_moment(coefficients, exponent)
