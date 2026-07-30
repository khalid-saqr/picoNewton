from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from piconewton_v3 import EndothelialControlVolume, FluidProperties
from piconewton_v3.hydrodynamics import WomersleySolver
from scipy.interpolate import BarycentricInterpolator

_EPS = 1e-30


@dataclass(frozen=True)
class AnalysisConfig:
    """Numerical settings for the waveform-susceptibility calculations."""

    radial_order: int = 150
    time_points: int = 2048
    quadrature_nodes: int = 256
    validation_epsilon: float = 0.08
    harmonics: int = 6

    def validate(self) -> None:
        if self.radial_order < 30:
            raise ValueError("radial_order must be at least 30")
        if self.time_points < 64:
            raise ValueError("time_points must be at least 64")
        if self.quadrature_nodes < 8:
            raise ValueError("quadrature_nodes must be at least 8")
        if not 0.0 < self.validation_epsilon <= 0.1:
            raise ValueError("validation_epsilon must lie in (0, 0.1]")
        if self.harmonics != 6:
            raise ValueError("the published model uses six input harmonics")


@dataclass(frozen=True)
class HarmonicHierarchy:
    radial_nodes: np.ndarray
    axial_base: np.ndarray
    azimuthal_first: np.ndarray
    axial_second: np.ndarray
    axial_vorticity_first: np.ndarray
    azimuthal_vorticity_base: np.ndarray
    azimuthal_vorticity_second: np.ndarray
    maximum_residual: float


@dataclass(frozen=True)
class ResponseBasis:
    radial_nodes: np.ndarray
    fields: Mapping[str, np.ndarray]
    maximum_residual: float


@dataclass(frozen=True)
class SusceptibilityResult:
    output_frequencies: np.ndarray
    spectrum: np.ndarray
    waveform: np.ndarray
    dimensionless_waveform: np.ndarray
    rms: float
    peak_absolute: float
    outward_duty: float
    inward_duty: float


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), _EPS))


def alpha_for_case(case: Any) -> float:
    fluid = FluidProperties()
    return float(
        case.radius_m
        * np.sqrt(fluid.angular_frequency_rad_s / fluid.kinematic_viscosity_m2_s)
    )


def eta_for_case(case: Any) -> float:
    return float(EndothelialControlVolume().thickness_m / case.radius_m)


def force_scale(case: Any) -> float:
    fluid = FluidProperties()
    endothelium = EndothelialControlVolume()
    velocity_scale = (
        case.pressure_gradient_scale_pa_per_m
        * case.radius_m**2
        / fluid.dynamic_viscosity_pa_s
    )
    return float(endothelium.area_m2 * fluid.density_kg_m3 * velocity_scale**2)


def _normalised_residual(matrix: np.ndarray, solution: np.ndarray, rhs: np.ndarray) -> float:
    residual = matrix @ solution - rhs
    denominator = (
        np.linalg.norm(matrix, np.inf) * np.linalg.norm(solution, np.inf)
        + np.linalg.norm(rhs, np.inf)
    )
    return float(np.linalg.norm(residual, np.inf) / max(denominator, _EPS))


def _solve_hierarchy_harmonic(
    solver: WomersleySolver,
    alpha: float,
    harmonic: int,
    forcing: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    axial_base, azimuthal_zero, base_residual = solver.solve_harmonic(
        alpha, harmonic, forcing, 0.0, 0.0, 1.0
    )
    if np.max(np.abs(azimuthal_zero)) > 1e-12:
        raise RuntimeError("isotropic azimuthal velocity is not zero")

    identity = sp.eye(solver.n, format="csr")
    azimuthal_matrix = (
        (1j * harmonic * alpha**2) * identity - solver.L1
    ).toarray()
    azimuthal_rhs = np.asarray(solver.L0 @ axial_base, dtype=complex)
    azimuthal_matrix[0, :] = 0.0
    azimuthal_matrix[0, 0] = 1.0
    azimuthal_matrix[-1, :] = 0.0
    azimuthal_matrix[-1, -1] = 1.0
    azimuthal_rhs[[0, -1]] = 0.0
    azimuthal_first = np.linalg.solve(azimuthal_matrix, azimuthal_rhs)
    azimuthal_residual = _normalised_residual(
        azimuthal_matrix, azimuthal_first, azimuthal_rhs
    )

    axial_matrix = ((1j * harmonic * alpha**2) * identity - solver.L0).toarray()
    axial_rhs = np.asarray(solver.L1 @ azimuthal_first, dtype=complex)
    axial_matrix[0, :] = solver.D[0, :]
    axial_matrix[-1, :] = 0.0
    axial_matrix[-1, -1] = 1.0
    axial_rhs[[0, -1]] = 0.0
    axial_second = np.linalg.solve(axial_matrix, axial_rhs)
    axial_residual = _normalised_residual(axial_matrix, axial_second, axial_rhs)

    return (
        axial_base,
        azimuthal_first,
        axial_second,
        max(base_residual, azimuthal_residual, axial_residual),
    )


def _vorticity_columns(
    solver: WomersleySolver,
    axial: np.ndarray,
    azimuthal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    axial_vorticity: list[np.ndarray] = []
    azimuthal_vorticity: list[np.ndarray] = []
    for column in range(axial.shape[1]):
        omega_z, omega_theta = solver.vorticity(axial[:, column], azimuthal[:, column])
        axial_vorticity.append(omega_z)
        azimuthal_vorticity.append(omega_theta)
    return np.stack(axial_vorticity, axis=1), np.stack(azimuthal_vorticity, axis=1)


def derive_hierarchy(case: Any, config: AnalysisConfig = AnalysisConfig()) -> HarmonicHierarchy:
    """Derive the reciprocal weak-anisotropy hierarchy for one arterial case."""

    config.validate()
    solver = WomersleySolver(config.radial_order, "verified")
    axial_base_columns: list[np.ndarray] = []
    azimuthal_first_columns: list[np.ndarray] = []
    axial_second_columns: list[np.ndarray] = []
    residuals: list[float] = []

    for harmonic, forcing in enumerate(
        case.harmonic_coefficients[: config.harmonics], start=1
    ):
        axial_base, azimuthal_first, axial_second, residual = (
            _solve_hierarchy_harmonic(
                solver,
                alpha_for_case(case),
                harmonic,
                complex(forcing),
            )
        )
        axial_base_columns.append(axial_base)
        azimuthal_first_columns.append(azimuthal_first)
        axial_second_columns.append(axial_second)
        residuals.append(residual)

    axial_base = np.stack(axial_base_columns, axis=1)
    azimuthal_first = np.stack(azimuthal_first_columns, axis=1)
    axial_second = np.stack(axial_second_columns, axis=1)
    axial_vorticity_first, _ = _vorticity_columns(
        solver, np.zeros_like(azimuthal_first), azimuthal_first
    )
    _, azimuthal_vorticity_base = _vorticity_columns(
        solver, axial_base, np.zeros_like(axial_base)
    )
    _, azimuthal_vorticity_second = _vorticity_columns(
        solver, axial_second, np.zeros_like(axial_second)
    )

    return HarmonicHierarchy(
        radial_nodes=solver.r.copy(),
        axial_base=axial_base,
        azimuthal_first=azimuthal_first,
        axial_second=axial_second,
        axial_vorticity_first=axial_vorticity_first,
        azimuthal_vorticity_base=azimuthal_vorticity_base,
        azimuthal_vorticity_second=azimuthal_vorticity_second,
        maximum_residual=float(max(residuals)),
    )


def unit_perturbation_response(
    case: Any, config: AnalysisConfig = AnalysisConfig()
) -> ResponseBasis:
    unit_case = replace(case, harmonic_coefficients=(1.0,) * config.harmonics)
    hierarchy = derive_hierarchy(unit_case, config)
    return ResponseBasis(
        radial_nodes=hierarchy.radial_nodes,
        fields={
            "axial_base": hierarchy.axial_base,
            "azimuthal_first": hierarchy.azimuthal_first,
            "axial_second": hierarchy.axial_second,
            "axial_vorticity_first": hierarchy.axial_vorticity_first,
            "azimuthal_vorticity_base": hierarchy.azimuthal_vorticity_base,
            "azimuthal_vorticity_second": hierarchy.azimuthal_vorticity_second,
        },
        maximum_residual=hierarchy.maximum_residual,
    )


def unit_full_response(
    case: Any,
    beta: float,
    gamma: float,
    delta: float,
    config: AnalysisConfig = AnalysisConfig(),
) -> ResponseBasis:
    config.validate()
    solver = WomersleySolver(config.radial_order, "verified")
    fields: dict[str, list[np.ndarray]] = {
        "axial": [],
        "azimuthal": [],
        "axial_vorticity": [],
        "azimuthal_vorticity": [],
    }
    residuals: list[float] = []
    for harmonic in range(1, config.harmonics + 1):
        axial, azimuthal, residual = solver.solve_harmonic(
            alpha_for_case(case), harmonic, 1.0, beta, gamma, delta
        )
        omega_z, omega_theta = solver.vorticity(axial, azimuthal)
        for name, value in zip(
            fields,
            (axial, azimuthal, omega_z, omega_theta),
            strict=True,
        ):
            fields[name].append(value)
        residuals.append(residual)
    return ResponseBasis(
        radial_nodes=solver.r.copy(),
        fields={name: np.stack(values, axis=1) for name, values in fields.items()},
        maximum_residual=float(max(residuals)),
    )


def _interpolate_columns(
    radial_nodes: np.ndarray, values: np.ndarray, query: np.ndarray
) -> np.ndarray:
    return np.stack(
        [
            BarycentricInterpolator(radial_nodes, values[:, column])(query)
            for column in range(values.shape[1])
        ],
        axis=1,
    )


def near_wall_basis(
    case: Any,
    basis: ResponseBasis,
    config: AnalysisConfig = AnalysisConfig(),
    eta: float | None = None,
) -> ResponseBasis:
    eta_value = eta_for_case(case) if eta is None else float(eta)
    if not 0.0 < eta_value < 1.0:
        raise ValueError("eta must lie in (0, 1)")
    radial = np.linspace(1.0 - eta_value, 1.0, config.quadrature_nodes)
    return ResponseBasis(
        radial_nodes=radial,
        fields={
            name: _interpolate_columns(basis.radial_nodes, values, radial)
            for name, values in basis.fields.items()
        },
        maximum_residual=basis.maximum_residual,
    )


def frequency_axis(harmonics: int) -> np.ndarray:
    return np.arange(-harmonics, harmonics + 1, dtype=int)


def canonical_coefficients(
    one_sided: Sequence[complex],
    phases_rad: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    one_sided_array = np.asarray(one_sided, dtype=complex)
    harmonics = len(one_sided_array)
    if phases_rad is not None:
        phases = np.asarray(phases_rad, dtype=float)
        if phases.shape != (harmonics,):
            raise ValueError("phase vector does not match the coefficient vector")
        one_sided_array = one_sided_array * np.exp(1j * phases)
    frequencies = frequency_axis(harmonics)
    coefficients = np.zeros(2 * harmonics + 1, dtype=complex)
    for harmonic in range(1, harmonics + 1):
        positive = one_sided_array[harmonic - 1] / 2.0
        coefficients[harmonics + harmonic] = positive
        coefficients[harmonics - harmonic] = np.conj(positive)
    return frequencies, coefficients


def _two_sided_response(one_sided: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    one_sided = np.asarray(one_sided, dtype=complex)
    harmonics = one_sided.shape[1]
    frequencies = frequency_axis(harmonics)
    response = np.zeros((one_sided.shape[0], 2 * harmonics + 1), dtype=complex)
    for harmonic in range(1, harmonics + 1):
        response[:, harmonics + harmonic] = one_sided[:, harmonic - 1]
        response[:, harmonics - harmonic] = np.conj(one_sided[:, harmonic - 1])
    return frequencies, response


def _two_sided_fields(basis: ResponseBasis) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frequencies: np.ndarray | None = None
    fields: dict[str, np.ndarray] = {}
    for name, values in basis.fields.items():
        current_frequencies, current_values = _two_sided_response(values)
        if frequencies is None:
            frequencies = current_frequencies
        fields[name] = current_values
    if frequencies is None:
        raise RuntimeError("response basis is empty")
    return frequencies, fields


def second_order_kernel(case: Any, basis: ResponseBasis) -> tuple[np.ndarray, np.ndarray]:
    frequencies, fields = _two_sided_fields(basis)
    kernel = np.zeros((len(frequencies), len(frequencies)), dtype=complex)
    scale = force_scale(case)
    for first_index in range(len(frequencies)):
        for second_index in range(len(frequencies)):
            integrand = (
                fields["azimuthal_first"][:, first_index]
                * fields["axial_vorticity_first"][:, second_index]
                - fields["axial_second"][:, first_index]
                * fields["azimuthal_vorticity_base"][:, second_index]
                - fields["axial_base"][:, first_index]
                * fields["azimuthal_vorticity_second"][:, second_index]
            )
            kernel[first_index, second_index] = scale * np.trapezoid(
                integrand, basis.radial_nodes
            )
    return frequencies, kernel


def full_force_kernel(case: Any, basis: ResponseBasis) -> tuple[np.ndarray, np.ndarray]:
    frequencies, fields = _two_sided_fields(basis)
    kernel = np.zeros((len(frequencies), len(frequencies)), dtype=complex)
    scale = force_scale(case)
    for first_index in range(len(frequencies)):
        for second_index in range(len(frequencies)):
            integrand = (
                fields["azimuthal"][:, first_index]
                * fields["axial_vorticity"][:, second_index]
                - fields["axial"][:, first_index]
                * fields["azimuthal_vorticity"][:, second_index]
            )
            kernel[first_index, second_index] = scale * np.trapezoid(
                integrand, basis.radial_nodes
            )
    return frequencies, kernel


def exact_excess_kernel(
    case: Any,
    beta: float,
    gamma: float,
    delta: float,
    config: AnalysisConfig = AnalysisConfig(),
    eta: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    anisotropic = near_wall_basis(
        case, unit_full_response(case, beta, gamma, delta, config), config, eta
    )
    isotropic = near_wall_basis(
        case, unit_full_response(case, 0.0, 0.0, delta, config), config, eta
    )
    frequencies, anisotropic_kernel = full_force_kernel(case, anisotropic)
    isotropic_frequencies, isotropic_kernel = full_force_kernel(case, isotropic)
    if not np.array_equal(frequencies, isotropic_frequencies):
        raise RuntimeError("frequency axes disagree")
    return (
        frequencies,
        anisotropic_kernel - isotropic_kernel,
        max(anisotropic.maximum_residual, isotropic.maximum_residual),
    )


def evaluate_kernel(
    frequencies: np.ndarray,
    kernel: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = np.asarray(frequencies, dtype=int)
    coefficients = np.asarray(coefficients, dtype=complex)
    output_frequencies = np.arange(
        2 * frequencies.min(), 2 * frequencies.max() + 1, dtype=int
    )
    spectrum = np.zeros(len(output_frequencies), dtype=complex)
    ordered = np.zeros_like(kernel, dtype=complex)
    output_minimum = int(output_frequencies[0])
    for first_index, first_frequency in enumerate(frequencies):
        for second_index, second_frequency in enumerate(frequencies):
            contribution = (
                kernel[first_index, second_index]
                * coefficients[first_index]
                * coefficients[second_index]
            )
            ordered[first_index, second_index] = contribution
            spectrum[int(first_frequency + second_frequency) - output_minimum] += contribution
    return output_frequencies, spectrum, ordered


def reconstruct_spectrum(
    output_frequencies: np.ndarray,
    spectrum: np.ndarray,
    time_points: int,
) -> np.ndarray:
    time_cycle = np.arange(time_points, dtype=float) / time_points
    basis = np.exp(1j * 2.0 * np.pi * np.outer(output_frequencies, time_cycle))
    return np.asarray(spectrum) @ basis


def susceptibility_from_kernel(
    case: Any,
    frequencies: np.ndarray,
    kernel: np.ndarray,
    one_sided_coefficients: Sequence[complex],
    config: AnalysisConfig = AnalysisConfig(),
    phases_rad: Sequence[float] | None = None,
) -> SusceptibilityResult:
    coefficient_frequencies, coefficients = canonical_coefficients(
        one_sided_coefficients, phases_rad
    )
    if not np.array_equal(frequencies, coefficient_frequencies):
        raise RuntimeError("coefficient and response frequency axes disagree")
    output_frequencies, spectrum, _ordered = evaluate_kernel(
        frequencies, kernel, coefficients
    )
    waveform = np.real(
        reconstruct_spectrum(output_frequencies, spectrum, config.time_points)
    )
    dimensionless = waveform / force_scale(case)
    return SusceptibilityResult(
        output_frequencies=output_frequencies,
        spectrum=spectrum / force_scale(case),
        waveform=waveform,
        dimensionless_waveform=dimensionless,
        rms=rms(dimensionless),
        peak_absolute=float(np.max(np.abs(dimensionless))),
        outward_duty=float(np.mean(dimensionless > 0.0)),
        inward_duty=float(np.mean(dimensionless < 0.0)),
    )


def critical_anisotropy(target_force_n: float, coefficient_n_per_epsilon2: float) -> float:
    if target_force_n <= 0.0:
        raise ValueError("target_force_n must be positive")
    if coefficient_n_per_epsilon2 <= 0.0:
        raise ValueError("coefficient_n_per_epsilon2 must be positive")
    return float(np.sqrt(target_force_n / coefficient_n_per_epsilon2))


def combine_unordered_pairs(
    frequencies: np.ndarray, ordered: np.ndarray
) -> list[dict[str, complex | int]]:
    rows: list[dict[str, complex | int]] = []
    for first_index, first_frequency in enumerate(frequencies):
        if first_frequency == 0:
            continue
        for second_index in range(first_index, len(frequencies)):
            second_frequency = int(frequencies[second_index])
            if second_frequency == 0:
                continue
            contribution = ordered[first_index, second_index]
            if first_index != second_index:
                contribution += ordered[second_index, first_index]
            rows.append(
                {
                    "m": int(first_frequency),
                    "n": second_frequency,
                    "q": int(first_frequency + second_frequency),
                    "contribution": contribution,
                }
            )
    return rows
