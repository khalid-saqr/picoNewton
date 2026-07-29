from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from piconewton_v3 import FluidProperties
from piconewton_v3.hydrodynamics import WomersleySolver
import scipy.sparse as sp

_EPS = 1e-30


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), _EPS))


@dataclass(frozen=True)
class Step4Config:
    profile: str = "publication"
    radial_order: int = 150
    time_points: int = 2048
    quadrature_nodes: int = 256
    epsilon_values: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10)
    slope_epsilon_max: float = 0.04
    parity_epsilon: float = 0.08
    relative_error_limit: float = 0.01
    minimum_valid_epsilon: float = 0.04

    def validate(self) -> None:
        if self.profile not in {"quick", "publication"}:
            raise ValueError("profile must be quick or publication")
        if self.radial_order < 30 or self.time_points < 64 or self.quadrature_nodes < 8:
            raise ValueError("invalid numerical resolution")
        eps = np.asarray(self.epsilon_values, dtype=float)
        if eps.ndim != 1 or len(eps) < 4 or np.any(~np.isfinite(eps)):
            raise ValueError("epsilon_values must contain at least four finite values")
        if np.any(eps <= 0.0) or np.any(eps > 0.1) or np.any(np.diff(eps) <= 0.0):
            raise ValueError("epsilon_values must be increasing in (0, 0.1]")
        if self.parity_epsilon <= 0.0 or self.parity_epsilon > 0.1:
            raise ValueError("parity_epsilon must lie in (0, 0.1]")
        if self.minimum_valid_epsilon not in self.epsilon_values:
            raise ValueError("minimum_valid_epsilon must be in epsilon_values")
        if not 0.0 < self.relative_error_limit < 1.0:
            raise ValueError("relative_error_limit must lie in (0,1)")


@dataclass(frozen=True)
class HarmonicHierarchy:
    r: np.ndarray
    uz0: np.ndarray
    ut1: np.ndarray
    uz2: np.ndarray
    oz1: np.ndarray
    ot0: np.ndarray
    ot2: np.ndarray
    max_residual: float


def _normalized_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    residual = A @ x - b
    return float(
        np.linalg.norm(residual, np.inf)
        / max(
            np.linalg.norm(A, np.inf) * np.linalg.norm(x, np.inf)
            + np.linalg.norm(b, np.inf),
            _EPS,
        )
    )


def _solve_scalar_hierarchy(
    solver: WomersleySolver,
    alpha: float,
    harmonic: int,
    forcing: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    uz0, ut0, residual0 = solver.solve_harmonic(
        alpha, harmonic, forcing, 0.0, 0.0, 1.0
    )
    if np.max(np.abs(ut0)) > 1e-12:
        raise RuntimeError("isotropic azimuthal field is not zero")

    identity = sp.eye(solver.n, format="csr")
    Atheta = ((1j * harmonic * alpha**2) * identity - solver.L1).toarray()
    rhs_theta = np.asarray(solver.L0 @ uz0, dtype=complex)
    Atheta[0, :] = 0.0
    Atheta[0, 0] = 1.0
    Atheta[-1, :] = 0.0
    Atheta[-1, -1] = 1.0
    rhs_theta[[0, -1]] = 0.0
    ut1 = np.linalg.solve(Atheta, rhs_theta)
    residual1 = _normalized_residual(Atheta, ut1, rhs_theta)

    Aaxial = ((1j * harmonic * alpha**2) * identity - solver.L0).toarray()
    rhs_axial = np.asarray(solver.L1 @ ut1, dtype=complex)
    Aaxial[0, :] = solver.D[0, :]
    Aaxial[-1, :] = 0.0
    Aaxial[-1, -1] = 1.0
    rhs_axial[[0, -1]] = 0.0
    uz2 = np.linalg.solve(Aaxial, rhs_axial)
    residual2 = _normalized_residual(Aaxial, uz2, rhs_axial)
    return uz0, ut1, uz2, max(residual0, residual1, residual2)


def _vorticity_columns(
    solver: WomersleySolver,
    axial: np.ndarray,
    azimuthal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    oz_columns: list[np.ndarray] = []
    ot_columns: list[np.ndarray] = []
    for column in range(axial.shape[1]):
        oz, ot = solver.vorticity(axial[:, column], azimuthal[:, column])
        oz_columns.append(oz)
        ot_columns.append(ot)
    return np.stack(oz_columns, axis=1), np.stack(ot_columns, axis=1)


def derive_hierarchy(case: Any, config: Step4Config) -> HarmonicHierarchy:
    fluid = FluidProperties()
    alpha = case.radius_m * np.sqrt(
        fluid.angular_frequency_rad_s / fluid.kinematic_viscosity_m2_s
    )
    solver = WomersleySolver(config.radial_order, "verified")
    uz0_columns: list[np.ndarray] = []
    ut1_columns: list[np.ndarray] = []
    uz2_columns: list[np.ndarray] = []
    residuals: list[float] = []
    for harmonic, forcing in enumerate(case.harmonic_coefficients, start=1):
        uz0, ut1, uz2, residual = _solve_scalar_hierarchy(
            solver, alpha, harmonic, forcing
        )
        uz0_columns.append(uz0)
        ut1_columns.append(ut1)
        uz2_columns.append(uz2)
        residuals.append(residual)
    uz0 = np.stack(uz0_columns, axis=1)
    ut1 = np.stack(ut1_columns, axis=1)
    uz2 = np.stack(uz2_columns, axis=1)
    oz1, _ = _vorticity_columns(solver, np.zeros_like(ut1), ut1)
    _, ot0 = _vorticity_columns(solver, uz0, np.zeros_like(uz0))
    _, ot2 = _vorticity_columns(solver, uz2, np.zeros_like(uz2))
    return HarmonicHierarchy(
        r=solver.r.copy(),
        uz0=uz0,
        ut1=ut1,
        uz2=uz2,
        oz1=oz1,
        ot0=ot0,
        ot2=ot2,
        max_residual=float(max(residuals)),
    )


def full_harmonic_fields(
    case: Any, config: Step4Config, epsilon: float
) -> tuple[dict[str, np.ndarray], float]:
    fluid = FluidProperties()
    alpha = case.radius_m * np.sqrt(
        fluid.angular_frequency_rad_s / fluid.kinematic_viscosity_m2_s
    )
    solver = WomersleySolver(config.radial_order, "verified")
    fields: dict[str, list[np.ndarray]] = {"uz": [], "ut": [], "oz": [], "ot": []}
    residuals: list[float] = []
    for harmonic, forcing in enumerate(case.harmonic_coefficients, start=1):
        uz, ut, residual = solver.solve_harmonic(
            alpha, harmonic, forcing, epsilon, epsilon, 1.0
        )
        oz, ot = solver.vorticity(uz, ut)
        for key, value in zip(fields, (uz, ut, oz, ot), strict=True):
            fields[key].append(value)
        residuals.append(residual)
    stacked = {key: np.stack(value, axis=1) for key, value in fields.items()}
    return stacked, float(max(residuals))
