from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from piconewton_v3 import EndothelialControlVolume, FluidProperties
from piconewton_v3.hydrodynamics import WomersleySolver
from scipy.interpolate import BarycentricInterpolator

from .perturbation_core import Step4Config, derive_hierarchy

_EPS = 1e-30


@dataclass(frozen=True)
class Step5Config:
    profile: str = "publication"
    radial_order: int = 150
    time_points: int = 2048
    quadrature_nodes: int = 256
    exact_epsilon: float = 0.1
    closure_tolerance: float = 1e-11
    selection_tolerance: float = 1e-12
    synthetic_phases_rad: tuple[float, ...] = (0.0, 0.37, -0.52, 1.10, -0.80, 0.25)

    def validate(self) -> None:
        if self.profile not in {"quick", "publication"}:
            raise ValueError("profile must be quick or publication")
        if self.radial_order < 30 or self.time_points < 64 or self.quadrature_nodes < 8:
            raise ValueError("invalid numerical resolution")
        if not 0.0 < abs(self.exact_epsilon) <= 0.1:
            raise ValueError("exact_epsilon must lie in [-0.1,0.1] excluding zero")
        if not 0.0 < self.closure_tolerance < 1.0:
            raise ValueError("closure_tolerance must lie in (0,1)")
        if not 0.0 < self.selection_tolerance < 1.0:
            raise ValueError("selection_tolerance must lie in (0,1)")
        if len(self.synthetic_phases_rad) != 6:
            raise ValueError("six synthetic phases are required")


@dataclass(frozen=True)
class ResponseBasis:
    radial_nodes: np.ndarray
    fields: Mapping[str, np.ndarray]
    max_residual: float

    @property
    def harmonics(self) -> int:
        first = next(iter(self.fields.values()))
        return int(first.shape[1])


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), _EPS))


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
    case: Any, basis: ResponseBasis, config: Step5Config
) -> tuple[np.ndarray, ResponseBasis]:
    endothelium = EndothelialControlVolume()
    eta = endothelium.thickness_m / case.radius_m
    radial = np.linspace(1.0 - eta, 1.0, config.quadrature_nodes)
    fields = {
        name: _interpolate_columns(basis.radial_nodes, values, radial)
        for name, values in basis.fields.items()
    }
    return radial, ResponseBasis(radial, fields, basis.max_residual)


def _alpha(case: Any) -> float:
    fluid = FluidProperties()
    return float(
        case.radius_m
        * np.sqrt(fluid.angular_frequency_rad_s / fluid.kinematic_viscosity_m2_s)
    )


def unit_full_response(case: Any, config: Step5Config, epsilon: float) -> ResponseBasis:
    solver = WomersleySolver(config.radial_order, "verified")
    fields: dict[str, list[np.ndarray]] = {"uz": [], "ut": [], "oz": [], "ot": []}
    residuals: list[float] = []
    for harmonic in range(1, 7):
        uz, ut, residual = solver.solve_harmonic(
            _alpha(case), harmonic, 1.0, epsilon, epsilon, 1.0
        )
        oz, ot = solver.vorticity(uz, ut)
        for name, value in zip(fields, (uz, ut, oz, ot), strict=True):
            fields[name].append(value)
        residuals.append(residual)
    return ResponseBasis(
        radial_nodes=solver.r.copy(),
        fields={name: np.stack(values, axis=1) for name, values in fields.items()},
        max_residual=float(max(residuals)),
    )


def unit_perturbation_response(case: Any, config: Step5Config) -> ResponseBasis:
    unit_case = replace(case, harmonic_coefficients=(1.0,) * 6)
    step4_config = Step4Config(
        profile=config.profile,
        radial_order=config.radial_order,
        time_points=config.time_points,
        quadrature_nodes=config.quadrature_nodes,
    )
    hierarchy = derive_hierarchy(unit_case, step4_config)
    return ResponseBasis(
        radial_nodes=hierarchy.r,
        fields={
            "uz0": hierarchy.uz0,
            "ut1": hierarchy.ut1,
            "uz2": hierarchy.uz2,
            "oz1": hierarchy.oz1,
            "ot0": hierarchy.ot0,
            "ot2": hierarchy.ot2,
        },
        max_residual=hierarchy.max_residual,
    )


def frequency_axis(harmonics: int) -> np.ndarray:
    return np.arange(-harmonics, harmonics + 1, dtype=int)


def canonical_coefficients(
    one_sided: Sequence[complex], phases_rad: Sequence[float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    one_sided_array = np.asarray(one_sided, dtype=complex)
    harmonics = len(one_sided_array)
    if phases_rad is not None:
        phase_array = np.asarray(phases_rad, dtype=float)
        if phase_array.shape != (harmonics,):
            raise ValueError("phase vector does not match the one-sided coefficients")
        one_sided_array = one_sided_array * np.exp(1j * phase_array)
    frequencies = frequency_axis(harmonics)
    coefficients = np.zeros(2 * harmonics + 1, dtype=complex)
    for harmonic in range(1, harmonics + 1):
        positive = one_sided_array[harmonic - 1] / 2.0
        coefficients[harmonics + harmonic] = positive
        coefficients[harmonics - harmonic] = np.conj(positive)
    return frequencies, coefficients


def two_sided_response(one_sided: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    one_sided = np.asarray(one_sided, dtype=complex)
    harmonics = one_sided.shape[1]
    frequencies = frequency_axis(harmonics)
    response = np.zeros((one_sided.shape[0], 2 * harmonics + 1), dtype=complex)
    for harmonic in range(1, harmonics + 1):
        response[:, harmonics + harmonic] = one_sided[:, harmonic - 1]
        response[:, harmonics - harmonic] = np.conj(one_sided[:, harmonic - 1])
    return frequencies, response


def force_scale(case: Any) -> float:
    fluid = FluidProperties()
    endothelium = EndothelialControlVolume()
    velocity_scale = (
        case.pressure_gradient_scale_pa_per_m
        * case.radius_m**2
        / fluid.dynamic_viscosity_pa_s
    )
    return float(endothelium.area_m2 * fluid.density_kg_m3 * velocity_scale**2)


def _two_sided_fields(basis: ResponseBasis) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    result: dict[str, np.ndarray] = {}
    frequencies: np.ndarray | None = None
    for name, values in basis.fields.items():
        current_frequencies, current = two_sided_response(values)
        if frequencies is None:
            frequencies = current_frequencies
        result[name] = current
    assert frequencies is not None
    return frequencies, result


def full_force_kernel(case: Any, basis: ResponseBasis) -> tuple[np.ndarray, np.ndarray]:
    frequencies, fields = _two_sided_fields(basis)
    size = len(frequencies)
    kernel = np.zeros((size, size), dtype=complex)
    scale = force_scale(case)
    for i in range(size):
        for j in range(size):
            integrand = (
                fields["ut"][:, i] * fields["oz"][:, j]
                - fields["uz"][:, i] * fields["ot"][:, j]
            )
            kernel[i, j] = scale * np.trapezoid(integrand, basis.radial_nodes)
    return frequencies, kernel


def second_order_force_kernel(case: Any, basis: ResponseBasis) -> tuple[np.ndarray, np.ndarray]:
    frequencies, fields = _two_sided_fields(basis)
    size = len(frequencies)
    kernel = np.zeros((size, size), dtype=complex)
    scale = force_scale(case)
    for i in range(size):
        for j in range(size):
            integrand = (
                fields["ut1"][:, i] * fields["oz1"][:, j]
                - fields["uz2"][:, i] * fields["ot0"][:, j]
                - fields["uz0"][:, i] * fields["ot2"][:, j]
            )
            kernel[i, j] = scale * np.trapezoid(integrand, basis.radial_nodes)
    return frequencies, kernel


def evaluate_kernel(
    frequencies: np.ndarray, kernel: np.ndarray, coefficients: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = np.asarray(frequencies, dtype=int)
    coefficients = np.asarray(coefficients, dtype=complex)
    output_frequencies = np.arange(2 * frequencies.min(), 2 * frequencies.max() + 1, dtype=int)
    spectrum = np.zeros(len(output_frequencies), dtype=complex)
    ordered = np.zeros_like(kernel, dtype=complex)
    output_min = int(output_frequencies[0])
    for i, first in enumerate(frequencies):
        for j, second in enumerate(frequencies):
            contribution = kernel[i, j] * coefficients[i] * coefficients[j]
            ordered[i, j] = contribution
            spectrum[int(first + second) - output_min] += contribution
    return output_frequencies, spectrum, ordered


def reconstruct_spectrum(
    output_frequencies: np.ndarray, spectrum: np.ndarray, time_points: int
) -> np.ndarray:
    time_cycle = np.arange(time_points, dtype=float) / time_points
    basis = np.exp(1j * 2.0 * np.pi * np.outer(output_frequencies, time_cycle))
    waveform = np.asarray(spectrum) @ basis
    return waveform


def sampled_spectrum(waveform: np.ndarray, output_frequencies: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform)
    fft = np.fft.fft(waveform) / waveform.size
    return np.asarray([fft[int(q) % waveform.size] for q in output_frequencies])


def _real_one_sided_fields(
    basis: ResponseBasis,
    coefficients: Sequence[complex],
    time_points: int,
) -> dict[str, np.ndarray]:
    coefficients_array = np.asarray(coefficients, dtype=complex)
    harmonics = np.arange(1, len(coefficients_array) + 1)
    time_cycle = np.arange(time_points, dtype=float) / time_points
    temporal = np.exp(1j * 2.0 * np.pi * np.outer(harmonics, time_cycle))
    return {
        name: np.real((values * coefficients_array[None, :]) @ temporal)
        for name, values in basis.fields.items()
    }


def direct_full_waveform(
    case: Any,
    anisotropic: ResponseBasis,
    isotropic: ResponseBasis,
    coefficients: Sequence[complex],
    time_points: int,
) -> np.ndarray:
    aniso = _real_one_sided_fields(anisotropic, coefficients, time_points)
    iso = _real_one_sided_fields(isotropic, coefficients, time_points)
    lamb_aniso = aniso["ut"] * aniso["oz"] - aniso["uz"] * aniso["ot"]
    lamb_iso = iso["ut"] * iso["oz"] - iso["uz"] * iso["ot"]
    return force_scale(case) * np.trapezoid(
        lamb_aniso - lamb_iso, anisotropic.radial_nodes, axis=0
    )


def direct_second_order_waveform(
    case: Any,
    perturbation: ResponseBasis,
    coefficients: Sequence[complex],
    time_points: int,
) -> np.ndarray:
    real = _real_one_sided_fields(perturbation, coefficients, time_points)
    lamb2 = (
        real["ut1"] * real["oz1"]
        - real["uz2"] * real["ot0"]
        - real["uz0"] * real["ot2"]
    )
    return force_scale(case) * np.trapezoid(lamb2, perturbation.radial_nodes, axis=0)


def combine_unordered_pairs(
    frequencies: np.ndarray, ordered: np.ndarray
) -> list[dict[str, complex | int]]:
    rows: list[dict[str, complex | int]] = []
    for i, first in enumerate(frequencies):
        if first == 0:
            continue
        for j in range(i, len(frequencies)):
            second = int(frequencies[j])
            if second == 0:
                continue
            contribution = ordered[i, j]
            if i != j:
                contribution += ordered[j, i]
            rows.append(
                {
                    "m": int(first),
                    "n": second,
                    "q": int(first + second),
                    "contribution": contribution,
                }
            )
    return rows
