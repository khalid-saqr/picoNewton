from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from piconewton_v3 import EndothelialControlVolume
from scipy.interpolate import BarycentricInterpolator

from .kernel_core import (
    ResponseBasis,
    Step5Config,
    canonical_coefficients,
    evaluate_kernel,
    force_scale,
    full_force_kernel,
    reconstruct_spectrum,
    relative_l2,
    second_order_force_kernel,
    unit_full_response,
    unit_perturbation_response,
)

_EPS = 1e-30
DEFAULT_ETA_REFERENCE = 2.361111e-3


@dataclass(frozen=True)
class Step7Config:
    profile: str = "publication"
    radial_order: int = 150
    time_points: int = 2048
    quadrature_nodes: int = 256
    exact_epsilon: float = 0.08
    eta_reference: float = DEFAULT_ETA_REFERENCE
    phase_scrambles: int = 8
    random_seed: int = 20260730
    closure_tolerance: float = 1e-11
    exact_relative_limit: float = 0.01

    def validate(self) -> None:
        if self.profile not in {"quick", "publication"}:
            raise ValueError("profile must be quick or publication")
        if self.radial_order < 30 or self.time_points < 64 or self.quadrature_nodes < 8:
            raise ValueError("invalid numerical resolution")
        if not 0.0 < self.exact_epsilon <= 0.1:
            raise ValueError("exact_epsilon must lie in (0,0.1]")
        if not 0.0 < self.eta_reference < 1.0:
            raise ValueError("eta_reference must lie in (0,1)")
        if self.phase_scrambles < 4:
            raise ValueError("at least four phase scrambles are required")
        if not 0.0 < self.closure_tolerance < 1.0:
            raise ValueError("closure_tolerance must lie in (0,1)")
        if not 0.0 < self.exact_relative_limit < 1.0:
            raise ValueError("exact_relative_limit must lie in (0,1)")

    def step5(self) -> Step5Config:
        return Step5Config(
            profile=self.profile,
            radial_order=self.radial_order,
            time_points=self.time_points,
            quadrature_nodes=self.quadrature_nodes,
            exact_epsilon=self.exact_epsilon,
        )


@dataclass(frozen=True)
class VesselResponseSet:
    perturbation: ResponseBasis
    anisotropic: ResponseBasis
    isotropic: ResponseBasis

    @property
    def max_residual(self) -> float:
        return float(
            max(
                self.perturbation.max_residual,
                self.anisotropic.max_residual,
                self.isotropic.max_residual,
            )
        )


def native_eta(case: Any) -> float:
    return float(EndothelialControlVolume().thickness_m / case.radius_m)


def response_set(case: Any, config: Step7Config) -> VesselResponseSet:
    step5 = config.step5()
    return VesselResponseSet(
        perturbation=unit_perturbation_response(case, step5),
        anisotropic=unit_full_response(case, step5, config.exact_epsilon),
        isotropic=unit_full_response(case, step5, 0.0),
    )


def _interpolate_columns(radial: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            BarycentricInterpolator(radial, values[:, column])(query)
            for column in range(values.shape[1])
        ],
        axis=1,
    )


def near_wall_at_eta(basis: ResponseBasis, eta: float, quadrature_nodes: int) -> ResponseBasis:
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must lie in (0,1)")
    radial = np.linspace(1.0 - eta, 1.0, quadrature_nodes)
    return ResponseBasis(
        radial_nodes=radial,
        fields={
            name: _interpolate_columns(basis.radial_nodes, values, radial)
            for name, values in basis.fields.items()
        },
        max_residual=basis.max_residual,
    )


def dimensionless_kernels(
    case: Any,
    responses: VesselResponseSet,
    eta: float,
    config: Step7Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perturbation = near_wall_at_eta(responses.perturbation, eta, config.quadrature_nodes)
    anisotropic = near_wall_at_eta(responses.anisotropic, eta, config.quadrature_nodes)
    isotropic = near_wall_at_eta(responses.isotropic, eta, config.quadrature_nodes)
    frequencies, kernel2_n = second_order_force_kernel(case, perturbation)
    full_frequencies, anisotropic_n = full_force_kernel(case, anisotropic)
    iso_frequencies, isotropic_n = full_force_kernel(case, isotropic)
    if not (
        np.array_equal(frequencies, full_frequencies)
        and np.array_equal(frequencies, iso_frequencies)
    ):
        raise RuntimeError("kernel frequency axes disagree")
    scale = force_scale(case)
    return frequencies, kernel2_n / scale, (anisotropic_n - isotropic_n) / scale


def evaluate_susceptibility(
    frequencies: np.ndarray,
    kernel: np.ndarray,
    one_sided: Sequence[complex],
    time_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    canonical_frequencies, coefficients = canonical_coefficients(one_sided)
    if not np.array_equal(canonical_frequencies, frequencies):
        raise RuntimeError("coefficient and kernel frequency axes disagree")
    output_frequencies, spectrum, _ordered = evaluate_kernel(frequencies, kernel, coefficients)
    waveform = np.real(reconstruct_spectrum(output_frequencies, spectrum, time_points))
    return output_frequencies, spectrum, waveform


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def susceptibility_metrics(waveform: np.ndarray, spectrum: np.ndarray) -> dict[str, float]:
    waveform = np.asarray(waveform, dtype=float)
    spectrum = np.asarray(spectrum, dtype=complex)
    positive = np.maximum(waveform, 0.0)
    negative = np.minimum(waveform, 0.0)
    output_frequencies = np.arange(-(len(spectrum) // 2), len(spectrum) // 2 + 1)
    non_dc = output_frequencies != 0
    high = np.abs(output_frequencies) > 6
    power = np.abs(spectrum) ** 2
    return {
        "phi_rms": rms(waveform),
        "phi_peak_abs": float(np.max(np.abs(waveform))),
        "phi_mean": float(np.mean(waveform)),
        "phi_positive_rms": rms(positive),
        "phi_negative_rms": rms(negative),
        "outward_duty": float(np.mean(waveform > 0.0)),
        "inward_duty": float(np.mean(waveform < 0.0)),
        "high_harmonic_fraction": float(
            np.sum(power[high]) / max(np.sum(power[non_dc]), _EPS)
        ),
    }


def input_rms(coefficients: Sequence[complex]) -> float:
    coefficients = np.asarray(coefficients, dtype=complex)
    return float(np.sqrt(0.5 * np.sum(np.abs(coefficients) ** 2)))


def normalise_input_rms(coefficients: Sequence[complex], target: float = 1.0) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=complex)
    return coefficients * target / max(input_rms(coefficients), _EPS)


def additive_decomposition(matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=float)
    grand = float(np.mean(values))
    row_mean = np.mean(values, axis=1)
    column_mean = np.mean(values, axis=0)
    fitted = row_mean[:, None] + column_mean[None, :] - grand
    interaction = values - fitted
    total = float(np.sum((values - grand) ** 2))
    row_ss = float(values.shape[1] * np.sum((row_mean - grand) ** 2))
    column_ss = float(values.shape[0] * np.sum((column_mean - grand) ** 2))
    interaction_ss = float(np.sum(interaction**2))
    return {
        "grand_mean": grand,
        "ss_total": total,
        "vessel_fraction": row_ss / max(total, _EPS),
        "waveform_fraction": column_ss / max(total, _EPS),
        "interaction_fraction": interaction_ss / max(total, _EPS),
        "max_additive_relative_residual": float(
            np.max(np.abs(interaction)) / max(np.max(np.abs(values)), _EPS)
        ),
    }


def causal_waveform_families() -> list[tuple[str, str, np.ndarray]]:
    families: list[tuple[str, str, np.ndarray]] = []
    for harmonic in range(1, 7):
        coefficients = np.zeros(6, dtype=complex)
        coefficients[harmonic - 1] = 1.0
        families.append(
            (f"single_h{harmonic}", "single_tone", normalise_input_rms(coefficients))
        )
    for first in range(1, 7):
        for second in range(first + 1, 7):
            coefficients = np.zeros(6, dtype=complex)
            coefficients[first - 1] = 1.0
            coefficients[second - 1] = np.exp(1j * np.pi / 3.0)
            families.append(
                (
                    f"two_h{first}_h{second}",
                    "two_tone",
                    normalise_input_rms(coefficients),
                )
            )
    for name, harmonics in (
        ("three_123", (1, 2, 3)),
        ("three_135", (1, 3, 5)),
        ("three_246", (2, 4, 6)),
    ):
        coefficients = np.zeros(6, dtype=complex)
        for index, harmonic in enumerate(harmonics):
            coefficients[harmonic - 1] = np.exp(1j * index * np.pi / 4.0)
        families.append((name, "sparse_three_tone", normalise_input_rms(coefficients)))
    for slope in (0.0, 0.5, 1.0, 1.5, 2.0):
        coefficients = np.arange(1, 7, dtype=float) ** (-slope)
        families.append(
            (f"slope_{slope:.1f}", "spectral_slope", normalise_input_rms(coefficients))
        )
    return families


def exact_second_order_error(
    exact_waveform: np.ndarray,
    second_order_waveform: np.ndarray,
    epsilon: float,
) -> dict[str, float]:
    scaled_exact = np.asarray(exact_waveform) / epsilon**2
    second_order_waveform = np.asarray(second_order_waveform)
    return {
        "waveform_relative_l2": relative_l2(scaled_exact, second_order_waveform),
        "rms_relative_error": abs(rms(scaled_exact) - rms(second_order_waveform))
        / max(rms(second_order_waveform), _EPS),
    }
