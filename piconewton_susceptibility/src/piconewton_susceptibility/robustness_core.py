from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from piconewton_v3 import V2_ARTERY_CASES
from piconewton_v3.hydrodynamics import WomersleySolver
from scipy.interpolate import BarycentricInterpolator

from .reduction_core import kernel_scale, susceptibility_from_kernel

_EPS = 1e-30
_FREQUENCIES = np.concatenate((np.arange(-6, 0), np.arange(1, 7))).astype(int)


@dataclass(frozen=True)
class ConstitutivePath:
    path_id: str
    beta_ratio: float
    gamma_ratio: float
    delta: float


DEFAULT_PATHS = (
    ConstitutivePath("reciprocal", 1.0, 1.0, 1.0),
    ConstitutivePath("beta_low", 0.5, 1.0, 1.0),
    ConstitutivePath("gamma_low", 1.0, 0.5, 1.0),
    ConstitutivePath("gamma_only", 0.0, 1.0, 1.0),
    ConstitutivePath("beta_only", 1.0, 0.0, 1.0),
    ConstitutivePath("beta075_gamma125", 0.75, 1.25, 1.0),
    ConstitutivePath("beta125_gamma075", 1.25, 0.75, 1.0),
    ConstitutivePath("delta_low", 1.0, 1.0, 0.8),
    ConstitutivePath("delta_high", 1.0, 1.0, 1.2),
)


@dataclass(frozen=True)
class Step9Config:
    profile: str = "publication"
    radial_order: int = 150
    quadrature_nodes: int = 256
    exact_epsilon: float = 0.08
    eta_reference: float = 2.361111e-3
    eta_multipliers: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
    resolution_pairs: tuple[tuple[int, int], ...] = ((120, 192), (180, 384))
    finite_epsilon_error_max: float = 0.02
    shape_median_error_max: float = 0.05
    shape_maximum_error_max: float = 0.20
    rank_one_energy_min: float = 0.999
    eta_median_error_max: float = 0.05
    eta_p90_error_max: float = 0.12
    eta_maximum_error_max: float = 0.20
    resolution_change_max: float = 0.01
    residual_max: float = 1e-10

    def validate(self) -> None:
        if self.profile not in {"quick", "publication"}:
            raise ValueError("profile must be quick or publication")
        if self.radial_order < 30 or self.quadrature_nodes < 8:
            raise ValueError("invalid numerical resolution")
        if not 0.0 < self.exact_epsilon <= 0.10:
            raise ValueError("exact_epsilon must lie in (0,0.10]")
        if not 0.0 < self.eta_reference < 1.0:
            raise ValueError("eta_reference must lie in (0,1)")
        if any(not 0.0 < value < 2.0 for value in self.eta_multipliers):
            raise ValueError("eta multipliers must be positive and below two")
        if not self.resolution_pairs:
            raise ValueError("at least one resolution pair is required")
        for radial_order, quadrature_nodes in self.resolution_pairs:
            if radial_order < 30 or quadrature_nodes < 8:
                raise ValueError("invalid robustness resolution pair")


@dataclass(frozen=True)
class HierarchyBasis:
    radial_nodes: np.ndarray
    fields: dict[str, np.ndarray]
    max_residual: float


@dataclass(frozen=True)
class FullBasis:
    radial_nodes: np.ndarray
    fields: dict[str, np.ndarray]
    max_residual: float


def alpha_for_case(case: Any) -> float:
    density = 1060.0
    kinematic_viscosity = 3.5e-6
    frequency_hz = 1.2
    del density
    return float(
        case.radius_m
        * np.sqrt(2.0 * np.pi * frequency_hz / kinematic_viscosity)
    )


def native_eta(case: Any) -> float:
    return float(1e-5 / case.radius_m)


def _backward_error(matrix: np.ndarray, solution: np.ndarray, rhs: np.ndarray) -> float:
    residual = matrix @ solution - rhs
    denominator = (
        np.linalg.norm(matrix, np.inf) * np.linalg.norm(solution, np.inf)
        + np.linalg.norm(rhs, np.inf)
    )
    return float(np.linalg.norm(residual, np.inf) / max(denominator, _EPS))


def _scalar_matrix(
    solver: WomersleySolver,
    alpha: float,
    harmonic: int,
    field: str,
    delta: float = 1.0,
) -> np.ndarray:
    identity = np.eye(solver.n, dtype=complex)
    operator = solver.L0.toarray() if field == "axial" else delta * solver.L1.toarray()
    matrix = 1j * harmonic * alpha**2 * identity - operator
    if field == "axial":
        matrix[0, :] = solver.D[0, :]
    else:
        matrix[0, :] = 0.0
        matrix[0, 0] = 1.0
    matrix[-1, :] = 0.0
    matrix[-1, -1] = 1.0
    return matrix


def derive_general_hierarchy(
    case: Any,
    radial_order: int,
    delta: float,
) -> HierarchyBasis:
    solver = WomersleySolver(radial_order, "verified")
    alpha = alpha_for_case(case)
    fields: dict[str, list[np.ndarray]] = {
        "uz0": [],
        "ut1": [],
        "uz2": [],
        "oz1": [],
        "ot0": [],
        "ot2": [],
    }
    residuals: list[float] = []
    for harmonic in range(1, 7):
        axial_matrix = _scalar_matrix(solver, alpha, harmonic, "axial")
        azimuthal_matrix = _scalar_matrix(
            solver, alpha, harmonic, "azimuthal", delta
        )
        axial_rhs = np.ones(solver.n, dtype=complex)
        axial_rhs[0] = axial_rhs[-1] = 0.0
        uz0 = np.linalg.solve(axial_matrix, axial_rhs)
        azimuthal_rhs = np.asarray(solver.L0 @ uz0)
        azimuthal_rhs[0] = azimuthal_rhs[-1] = 0.0
        ut1 = np.linalg.solve(azimuthal_matrix, azimuthal_rhs)
        correction_rhs = np.asarray(solver.L1 @ ut1)
        correction_rhs[0] = correction_rhs[-1] = 0.0
        uz2 = np.linalg.solve(axial_matrix, correction_rhs)
        oz1, _ = solver.vorticity(np.zeros_like(ut1), ut1)
        ot0 = -(solver.D @ uz0)
        ot2 = -(solver.D @ uz2)
        values = (uz0, ut1, uz2, oz1, ot0, ot2)
        for name, value in zip(fields, values, strict=True):
            fields[name].append(value)
        residuals.extend(
            (
                _backward_error(axial_matrix, uz0, axial_rhs),
                _backward_error(azimuthal_matrix, ut1, azimuthal_rhs),
                _backward_error(axial_matrix, uz2, correction_rhs),
            )
        )
    return HierarchyBasis(
        radial_nodes=solver.r.copy(),
        fields={name: np.stack(values, axis=1) for name, values in fields.items()},
        max_residual=float(max(residuals)),
    )


def solve_full_response(
    case: Any,
    radial_order: int,
    beta: float,
    gamma: float,
    delta: float,
) -> FullBasis:
    solver = WomersleySolver(radial_order, "verified")
    alpha = alpha_for_case(case)
    identity = sp.eye(solver.n, format="csr")
    fields: dict[str, list[np.ndarray]] = {"uz": [], "ut": [], "oz": [], "ot": []}
    residuals: list[float] = []
    for harmonic in range(1, 7):
        a_zz = ((1j * harmonic * alpha**2) * identity - solver.L0).tolil()
        a_zt = (-beta * solver.L1).tolil()
        a_tz = (-gamma * solver.L0).tolil()
        a_tt = ((1j * harmonic * alpha**2) * identity - delta * solver.L1).tolil()
        b_z = np.ones(solver.n, dtype=complex)
        b_t = np.zeros(solver.n, dtype=complex)
        a_zz[0, :], a_zt[0, :], b_z[0] = solver.D[0, :], 0.0, 0.0
        a_tz[0, :], a_tt[0, :], a_tt[0, 0], b_t[0] = 0.0, 0.0, 1.0, 0.0
        a_zz[-1, :], a_zz[-1, -1], a_zt[-1, :], b_z[-1] = (
            0.0,
            1.0,
            0.0,
            0.0,
        )
        a_tz[-1, :], a_tt[-1, :], a_tt[-1, -1], b_t[-1] = (
            0.0,
            0.0,
            1.0,
            0.0,
        )
        matrix = sp.vstack(
            [
                sp.hstack([a_zz.tocsr(), a_zt.tocsr()]),
                sp.hstack([a_tz.tocsr(), a_tt.tocsr()]),
            ],
            format="csc",
        )
        rhs = np.concatenate([b_z, b_t])
        solution = spla.spsolve(matrix, rhs)
        uz = solution[: solver.n]
        ut = solution[solver.n :]
        oz, ot = solver.vorticity(uz, ut)
        for name, value in zip(fields, (uz, ut, oz, ot), strict=True):
            fields[name].append(value)
        residual = matrix @ solution - rhs
        matrix_norm = float(np.max(np.asarray(np.abs(matrix).sum(axis=1)).ravel()))
        denominator = matrix_norm * np.linalg.norm(solution, np.inf) + np.linalg.norm(
            rhs, np.inf
        )
        residuals.append(float(np.linalg.norm(residual, np.inf) / max(denominator, _EPS)))
    return FullBasis(
        radial_nodes=solver.r.copy(),
        fields={name: np.stack(values, axis=1) for name, values in fields.items()},
        max_residual=float(max(residuals)),
    )


def _interpolate_basis(
    radial_nodes: np.ndarray,
    fields: dict[str, np.ndarray],
    eta: float,
    quadrature_nodes: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    query = np.linspace(1.0 - eta, 1.0, quadrature_nodes)
    matrix = BarycentricInterpolator(radial_nodes, np.eye(len(radial_nodes)), axis=0)(query)
    return query, {name: matrix @ values for name, values in fields.items()}


def _two_sided(values: np.ndarray) -> np.ndarray:
    response = np.zeros((values.shape[0], 12), dtype=complex)
    for harmonic in range(1, 7):
        response[:, 6 + harmonic - 1] = values[:, harmonic - 1]
        response[:, 6 - harmonic] = np.conj(values[:, harmonic - 1])
    return response


def hierarchy_kernel(
    basis: HierarchyBasis,
    eta: float,
    quadrature_nodes: int,
    beta_ratio: float,
    gamma_ratio: float,
) -> np.ndarray:
    radial, interpolated = _interpolate_basis(
        basis.radial_nodes, basis.fields, eta, quadrature_nodes
    )
    fields = {name: _two_sided(values) for name, values in interpolated.items()}
    kernel = np.zeros((12, 12), dtype=complex)
    for first in range(12):
        for second in range(12):
            integrand = (
                gamma_ratio**2
                * fields["ut1"][:, first]
                * fields["oz1"][:, second]
                - beta_ratio
                * gamma_ratio
                * fields["uz2"][:, first]
                * fields["ot0"][:, second]
                - beta_ratio
                * gamma_ratio
                * fields["uz0"][:, first]
                * fields["ot2"][:, second]
            )
            kernel[first, second] = np.trapezoid(integrand, radial)
    return kernel


def full_excess_kernel(
    anisotropic: FullBasis,
    isotropic: FullBasis,
    eta: float,
    quadrature_nodes: int,
) -> np.ndarray:
    radial, anisotropic_fields = _interpolate_basis(
        anisotropic.radial_nodes, anisotropic.fields, eta, quadrature_nodes
    )
    _, isotropic_fields = _interpolate_basis(
        isotropic.radial_nodes, isotropic.fields, eta, quadrature_nodes
    )
    anisotropic_two = {
        name: _two_sided(values) for name, values in anisotropic_fields.items()
    }
    isotropic_two = {name: _two_sided(values) for name, values in isotropic_fields.items()}
    kernel = np.zeros((12, 12), dtype=complex)
    for first in range(12):
        for second in range(12):
            anisotropic_lamb = (
                anisotropic_two["ut"][:, first] * anisotropic_two["oz"][:, second]
                - anisotropic_two["uz"][:, first] * anisotropic_two["ot"][:, second]
            )
            isotropic_lamb = (
                isotropic_two["ut"][:, first] * isotropic_two["oz"][:, second]
                - isotropic_two["uz"][:, first] * isotropic_two["ot"][:, second]
            )
            kernel[first, second] = np.trapezoid(
                anisotropic_lamb - isotropic_lamb, radial
            )
    return kernel


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(reference))
        / max(np.linalg.norm(np.asarray(reference)), _EPS)
    )


def normalised_kernel_error(actual: np.ndarray, reference: np.ndarray) -> float:
    actual_scale = kernel_scale(actual)
    reference_scale = kernel_scale(reference)
    if actual_scale <= _EPS or reference_scale <= _EPS:
        return float("nan")
    return relative_l2(actual / actual_scale, reference / reference_scale)


def waveform_error(
    exact_kernel: np.ndarray,
    predicted_kernel: np.ndarray,
    coefficients: Sequence[complex],
) -> float:
    exact = susceptibility_from_kernel(exact_kernel, coefficients)
    predicted = susceptibility_from_kernel(predicted_kernel, coefficients)
    return abs(predicted - exact) / max(exact, _EPS)


def case_by_id(artery_id: str) -> Any:
    return next(case for case in V2_ARTERY_CASES if case.artery_id == artery_id)
