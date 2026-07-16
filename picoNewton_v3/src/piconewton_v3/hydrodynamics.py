"""Verified and reproduction anisotropic Womersley solvers."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import BarycentricInterpolator
from scipy.special import jv
from threadpoolctl import threadpool_limits

from .types import (
    ArteryCase, EndothelialControlVolume, FluidProperties, HydrodynamicConfig, SolverMode
)

_BLAS_LIMITER = threadpool_limits(limits=1, user_api="blas")

class WomersleySolver:
    """Chebyshev collocation solver for the anisotropic harmonic block system."""

    def __init__(self, radial_order: int, mode: SolverMode = "verified"):
        if radial_order < 30:
            raise ValueError("radial_order must be at least 30")
        if mode not in ("verified", "reproduction"):
            raise ValueError("unknown mode")
        self.radial_order = radial_order
        self.n = radial_order + 1
        self.mode = mode
        self._setup_discretization()

    def _setup_discretization(self) -> None:
        k = np.arange(self.n)
        x = np.cos(np.pi * k / self.radial_order)
        c = np.ones(self.n)
        c[0] = c[-1] = 2.0
        c *= (-1.0) ** k

        if self.mode == "verified":
            # Rows correspond to evaluation points. This orientation differentiates
            # polynomial test functions to machine precision after the x->r map.
            X = np.tile(x, (self.n, 1)).T
            dX = X - X.T
            D_x = np.outer(c, 1.0 / c) / (dX + np.eye(self.n))
        else:
            # Current public executable layout, retained only for traceability.
            X = np.tile(x, (self.n, 1))
            dX = X - X.T + np.eye(self.n)
            D_x = np.outer(c, 1.0 / c) / dX

        D_x -= np.diag(np.sum(D_x, axis=1))
        self.r = (1.0 - x) / 2.0
        self.D = -2.0 * D_x

        r_safe = self.r.copy()
        r_safe[0] = 1.0 if self.mode == "verified" else 1e-12
        D = sp.csr_matrix(self.D)
        D2 = sp.csr_matrix(self.D @ self.D)
        self.L0 = D2 + sp.diags(1.0 / r_safe) @ D
        self.L1 = self.L0 - sp.diags(1.0 / r_safe**2)

    def derivative_polynomial_error(self) -> float:
        exact = 2.0 * self.r
        return float(np.max(np.abs(self.D @ self.r**2 - exact)))

    def solve_harmonic(
        self,
        alpha: float,
        harmonic: int,
        forcing: complex,
        beta: float,
        gamma: float,
        delta: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if harmonic < 1:
            raise ValueError("harmonic must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if delta - ((beta + gamma) / 2.0) ** 2 <= 0:
            raise ValueError("constitutive sample violates positive dissipation")

        I = sp.eye(self.n, format="csr")
        Azz = ((1j * harmonic * alpha**2) * I - self.L0).tolil()
        Azt = (-beta * self.L1).tolil()
        Atz = (-gamma * self.L0).tolil()
        Att = ((1j * harmonic * alpha**2) * I - delta * self.L1).tolil()

        bz = forcing * np.ones(self.n, dtype=complex)
        bt = np.zeros(self.n, dtype=complex)

        # r=0: axial symmetry, azimuthal regularity.
        Azz[0, :], Azt[0, :], bz[0] = self.D[0, :], 0.0, 0.0
        Atz[0, :], Att[0, :], Att[0, 0], bt[0] = 0.0, 0.0, 1.0, 0.0
        # r=1: no slip for both components.
        Azz[-1, :], Azz[-1, -1], Azt[-1, :], bz[-1] = 0.0, 1.0, 0.0, 0.0
        Atz[-1, :], Att[-1, :], Att[-1, -1], bt[-1] = 0.0, 0.0, 1.0, 0.0

        A = sp.vstack(
            [
                sp.hstack([Azz.tocsr(), Azt.tocsr()]),
                sp.hstack([Atz.tocsr(), Att.tocsr()]),
            ],
            format="csc",
        )
        rhs = np.concatenate([bz, bt])
        if self.mode == "verified":
            # Dense LAPACK is predictable for these modest spectral systems and
            # avoids SuperLU pathologies observed for some admissible parameter
            # combinations in large coverage sweeps.
            A_evaluate = A.toarray()
            solution = np.linalg.solve(A_evaluate, rhs)
            residual = A_evaluate @ solution - rhs
            A_norm = float(np.linalg.norm(A_evaluate, np.inf))
        else:
            solution = spla.spsolve(A, rhs)
            residual = A @ solution - rhs
            A_norm = float(np.max(np.asarray(np.abs(A).sum(axis=1)).ravel()))
        x_norm = float(np.linalg.norm(solution, np.inf))
        b_norm = float(np.linalg.norm(rhs, np.inf))
        backward_residual = float(
            np.linalg.norm(residual, np.inf) / max(A_norm * x_norm + b_norm, 1e-30)
        )
        return solution[: self.n], solution[self.n :], backward_residual

    def vorticity(
        self, axial_velocity: np.ndarray, azimuthal_velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        omega_theta = -(self.D @ axial_velocity)
        d_r_u_theta = self.D @ (self.r * azimuthal_velocity)
        omega_z = np.empty_like(azimuthal_velocity)
        omega_z[1:] = d_r_u_theta[1:] / self.r[1:]
        # For regular u_theta ~ a*r, (1/r)d(r*u_theta)/dr -> 2a.
        omega_z[0] = 2.0 * (self.D @ azimuthal_velocity)[0]
        return omega_z, omega_theta


def classical_womersley_solution(
    r: np.ndarray, alpha: float, harmonic: int = 1, forcing: complex = 1.0
) -> np.ndarray:
    kappa = alpha * np.sqrt(-1j * harmonic)
    return forcing / (1j * harmonic * alpha**2) * (
        1.0 - jv(0, kappa * r) / jv(0, kappa)
    )


def isotropic_validation(
    alpha_values: Sequence[float] = (3.0, 5.0, 8.0, 12.0, 20.0),
    radial_order: int = 150,
) -> list[dict[str, float | bool]]:
    solver = WomersleySolver(radial_order, "verified")
    rows: list[dict[str, float | bool]] = []
    for alpha in alpha_values:
        uz, ut, residual = solver.solve_harmonic(alpha, 1, 1.0, 0.0, 0.0, 1.0)
        exact = classical_womersley_solution(solver.r, alpha)
        error = float(np.max(np.abs(uz - exact)))
        rows.append(
            {
                "alpha": float(alpha),
                "radial_order": radial_order,
                "linf_error": error,
                "max_abs_azimuthal_velocity": float(np.max(np.abs(ut))),
                "normalized_backward_residual": residual,
                "passed": bool(error < 1e-8 and residual < 1e-13),
            }
        )
    return rows


def _interpolate_columns(
    radial_nodes: np.ndarray, values: np.ndarray, query: np.ndarray
) -> np.ndarray:
    return np.stack(
        [BarycentricInterpolator(radial_nodes, values[:, j])(query) for j in range(values.shape[1])],
        axis=1,
    )


def compute_hydrodynamics(
    case: ArteryCase,
    hydro: HydrodynamicConfig,
    fluid: FluidProperties = FluidProperties(),
    endothelium: EndothelialControlVolume = EndothelialControlVolume(),
    *,
    phases_rad: Sequence[float] | None = None,
    harmonics_retained: int = 6,
    include_near_wall_fields: bool = False,
) -> dict[str, np.ndarray | float | str | int]:
    """Compute real fields, Lamb force and anisotropic axial wall shear traction.

    ``verified`` mode reconstructs real velocity and vorticity fields before the
    nonlinear product. ``reproduction`` mode intentionally retains the current
    public harmonic-product ordering for regression comparison only.
    """
    case.validate()
    hydro.validate()
    fluid.validate()
    endothelium.validate()
    if not 1 <= harmonics_retained <= len(case.harmonic_coefficients):
        raise ValueError("invalid harmonics_retained")
    if phases_rad is not None and len(phases_rad) < harmonics_retained:
        raise ValueError("phases_rad is shorter than harmonics_retained")

    alpha = case.radius_m * np.sqrt(
        fluid.angular_frequency_rad_s / fluid.kinematic_viscosity_m2_s
    )
    velocity_scale = (
        case.pressure_gradient_scale_pa_per_m
        * case.radius_m**2
        / fluid.dynamic_viscosity_pa_s
    )
    solver = WomersleySolver(hydro.radial_order, hydro.mode)

    fields: dict[str, list[np.ndarray]] = {"uz": [], "ut": [], "oz": [], "ot": []}
    residuals: list[float] = []
    for harmonic, coefficient in enumerate(
        case.harmonic_coefficients[:harmonics_retained], start=1
    ):
        phase = 1.0 if phases_rad is None else np.exp(1j * phases_rad[harmonic - 1])
        uz, ut, residual = solver.solve_harmonic(
            alpha,
            harmonic,
            coefficient,
            hydro.beta,
            hydro.gamma,
            hydro.delta,
        )
        oz, ot = solver.vorticity(uz, ut)
        for key, value in zip(fields, (uz, ut, oz, ot)):
            fields[key].append(value * phase)
        residuals.append(residual)
    harmonic_fields = {key: np.stack(value, axis=1) for key, value in fields.items()}

    time_cycle = np.arange(hydro.time_points, dtype=float) / hydro.time_points
    harmonic_numbers = np.arange(1, harmonics_retained + 1)
    time_basis = np.exp(1j * 2.0 * np.pi * np.outer(harmonic_numbers, time_cycle))

    epsilon = endothelium.thickness_m / case.radius_m
    if not 0.0 < epsilon < 1.0:
        raise ValueError("endothelial thickness must be smaller than the artery radius")
    near_wall_r = np.linspace(1.0 - epsilon, 1.0, hydro.quadrature_nodes)
    near_wall = {
        key: _interpolate_columns(solver.r, value, near_wall_r)
        for key, value in harmonic_fields.items()
    }

    if hydro.mode == "reproduction":
        lamb_h = near_wall["ut"] * near_wall["oz"] - near_wall["uz"] * near_wall["ot"]
        lamb_r_dimensionless = np.real(lamb_h @ time_basis)
        # Reconstructed fields are still returned for diagnostics.
        uz_real = np.real(near_wall["uz"] @ time_basis)
        ut_real = np.real(near_wall["ut"] @ time_basis)
        oz_real = np.real(near_wall["oz"] @ time_basis)
        ot_real = np.real(near_wall["ot"] @ time_basis)
    else:
        uz_real = np.real(near_wall["uz"] @ time_basis)
        ut_real = np.real(near_wall["ut"] @ time_basis)
        oz_real = np.real(near_wall["oz"] @ time_basis)
        ot_real = np.real(near_wall["ot"] @ time_basis)
        lamb_r_dimensionless = ut_real * oz_real - uz_real * ot_real

    # rho*ell has units N/m^3. Multiplying by A_EC and physical dr gives N.
    force_signed_n = (
        endothelium.area_m2
        * fluid.density_kg_m3
        * velocity_scale**2
        * np.trapezoid(lamb_r_dimensionless, near_wall_r, axis=0)
    )
    force_exposure_n = (
        endothelium.area_m2
        * fluid.density_kg_m3
        * velocity_scale**2
        * np.trapezoid(np.abs(lamb_r_dimensionless), near_wall_r, axis=0)
    )

    # Inherited anisotropic axial traction at r=R:
    # tau_zr = mu U0/R [dUz/dr* + beta(dUtheta/dr* - Utheta/r*)].
    d_uz = solver.D @ harmonic_fields["uz"]
    d_ut = solver.D @ harmonic_fields["ut"]
    shear_h = (
        d_uz[-1, :]
        + hydro.beta * (d_ut[-1, :] - harmonic_fields["ut"][-1, :])
    ) * (fluid.dynamic_viscosity_pa_s * velocity_scale / case.radius_m)
    wall_shear_pa = np.real(shear_h @ time_basis)

    result: dict[str, np.ndarray | float | str | int] = {
        "artery_id": case.artery_id,
        "artery_name": case.name,
        "solver_mode": hydro.mode,
        "radial_order": hydro.radial_order,
        "time_points": hydro.time_points,
        "quadrature_nodes": hydro.quadrature_nodes,
        "harmonics_retained": harmonics_retained,
        "beta": float(hydro.beta),
        "gamma": float(hydro.gamma),
        "delta": float(hydro.delta),
        "alpha": float(alpha),
        "velocity_scale_m_s": float(velocity_scale),
        "time_cycle": time_cycle,
        "time_s": time_cycle / fluid.fundamental_frequency_hz,
        "near_wall_r_star": near_wall_r,
        "force_signed_n": force_signed_n,
        "force_exposure_n": force_exposure_n,
        "wall_shear_pa": wall_shear_pa,
        "max_normalized_backward_residual": float(max(residuals)),
        "differentiation_polynomial_error": solver.derivative_polynomial_error(),
    }
    if include_near_wall_fields:
        result.update(
            {
                "u_z_m_s": uz_real * velocity_scale,
                "u_theta_m_s": ut_real * velocity_scale,
                "omega_z_s_inv": oz_real * velocity_scale / case.radius_m,
                "omega_theta_s_inv": ot_real * velocity_scale / case.radius_m,
                "lamb_r_m_s2": lamb_r_dimensionless * velocity_scale**2 / case.radius_m,
                "force_density_r_n_m3": lamb_r_dimensionless
                * fluid.density_kg_m3
                * velocity_scale**2
                / case.radius_m,
            }
        )
    return result
