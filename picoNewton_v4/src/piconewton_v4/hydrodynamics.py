"""Standalone anisotropic Womersley hydrodynamics used by picoNewton_v4."""
from __future__ import annotations
from dataclasses import asdict
from typing import Sequence
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BarycentricInterpolator
from scipy.special import jv
from threadpoolctl import threadpool_limits
from .types import ArteryCase, EndothelialControlVolume, FluidProperties, HydrodynamicConfig

_BLAS_LIMITER = threadpool_limits(limits=1, user_api="blas")

class WomersleySolver:
    def __init__(self, radial_order: int):
        if radial_order < 30:
            raise ValueError("radial_order must be at least 30")
        self.radial_order = int(radial_order)
        self.n = self.radial_order + 1
        self._setup()
    def _setup(self) -> None:
        k = np.arange(self.n)
        x = np.cos(np.pi * k / self.radial_order)
        c = np.ones(self.n); c[0] = c[-1] = 2.0; c *= (-1.0) ** k
        X = np.tile(x, (self.n, 1)).T
        dX = X - X.T
        D_x = np.outer(c, 1.0 / c) / (dX + np.eye(self.n))
        D_x -= np.diag(np.sum(D_x, axis=1))
        self.r = (1.0 - x) / 2.0
        self.D = -2.0 * D_x
        r_safe = self.r.copy(); r_safe[0] = 1.0
        D = sp.csr_matrix(self.D); D2 = sp.csr_matrix(self.D @ self.D)
        self.L0 = D2 + sp.diags(1.0 / r_safe) @ D
        self.L1 = self.L0 - sp.diags(1.0 / r_safe**2)
    def derivative_polynomial_error(self) -> float:
        return float(np.max(np.abs(self.D @ self.r**2 - 2.0 * self.r)))
    def solve_harmonic(self, *, alpha: float, harmonic: int, forcing: complex,
                       beta: float, gamma: float, delta: float) -> tuple[np.ndarray, np.ndarray, float]:
        if alpha <= 0 or harmonic < 1:
            raise ValueError("invalid harmonic problem")
        if delta - ((beta + gamma) / 2.0) ** 2 <= 0:
            raise ValueError("constitutive parameters violate positive dissipation")
        eye = sp.eye(self.n, format="csr")
        Azz = ((1j * harmonic * alpha**2) * eye - self.L0).tolil()
        Azt = (-beta * self.L1).tolil()
        Atz = (-gamma * self.L0).tolil()
        Att = ((1j * harmonic * alpha**2) * eye - delta * self.L1).tolil()
        bz = forcing * np.ones(self.n, dtype=complex); bt = np.zeros(self.n, dtype=complex)
        Azz[0, :], Azt[0, :], bz[0] = self.D[0, :], 0.0, 0.0
        Atz[0, :], Att[0, :], Att[0, 0], bt[0] = 0.0, 0.0, 1.0, 0.0
        Azz[-1, :], Azz[-1, -1], Azt[-1, :], bz[-1] = 0.0, 1.0, 0.0, 0.0
        Atz[-1, :], Att[-1, :], Att[-1, -1], bt[-1] = 0.0, 0.0, 1.0, 0.0
        A = sp.vstack([sp.hstack([Azz.tocsr(), Azt.tocsr()]), sp.hstack([Atz.tocsr(), Att.tocsr()])], format="csc").toarray()
        rhs = np.concatenate([bz, bt]); solution = np.linalg.solve(A, rhs)
        residual = A @ solution - rhs
        denominator = max(float(np.linalg.norm(A, np.inf)) * float(np.linalg.norm(solution, np.inf)) + float(np.linalg.norm(rhs, np.inf)), 1e-30)
        return solution[:self.n], solution[self.n:], float(np.linalg.norm(residual, np.inf) / denominator)
    def vorticity(self, axial_velocity: np.ndarray, azimuthal_velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        omega_theta = -(self.D @ axial_velocity)
        d_r_ut = self.D @ (self.r * azimuthal_velocity)
        omega_z = np.empty_like(azimuthal_velocity)
        omega_z[1:] = d_r_ut[1:] / self.r[1:]
        omega_z[0] = 2.0 * (self.D @ azimuthal_velocity)[0]
        return omega_z, omega_theta

def classical_womersley_solution(r: np.ndarray, alpha: float, harmonic: int = 1, forcing: complex = 1.0) -> np.ndarray:
    kappa = alpha * np.sqrt(-1j * harmonic)
    return forcing / (1j * harmonic * alpha**2) * (1.0 - jv(0, kappa * r) / jv(0, kappa))

def isotropic_validation(alpha_values: Sequence[float] = (3.0, 5.0, 8.0, 12.0, 20.0), radial_order: int = 80):
    solver = WomersleySolver(radial_order); rows = []
    for alpha in alpha_values:
        uz, ut, residual = solver.solve_harmonic(alpha=alpha, harmonic=1, forcing=1.0, beta=0.0, gamma=0.0, delta=1.0)
        error = float(np.max(np.abs(uz - classical_womersley_solution(solver.r, alpha))))
        rows.append({"alpha": float(alpha), "linf_error": error, "max_abs_azimuthal_velocity": float(np.max(np.abs(ut))), "normalized_backward_residual": residual})
    return rows

def _interpolate_columns(radial_nodes: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.asarray(BarycentricInterpolator(radial_nodes, values, axis=0)(query))

def _harmonic_fields_with_fluid(solver: WomersleySolver, case: ArteryCase, config: HydrodynamicConfig, fluid: FluidProperties):
    fields = {"uz": [], "ut": [], "oz": [], "ot": []}; residuals = []
    alpha = case.radius_m * np.sqrt(fluid.angular_frequency_rad_s / fluid.kinematic_viscosity_m2_s)
    for harmonic, coefficient in enumerate(case.harmonic_coefficients, start=1):
        uz, ut, residual = solver.solve_harmonic(alpha=alpha, harmonic=harmonic, forcing=coefficient,
                                                  beta=config.beta, gamma=config.gamma, delta=config.delta)
        oz, ot = solver.vorticity(uz, ut)
        for key, value in zip(("uz", "ut", "oz", "ot"), (uz, ut, oz, ot)):
            fields[key].append(value)
        residuals.append(residual)
    return {key: np.stack(value, axis=1) for key, value in fields.items()}, residuals

def compute_case(case: ArteryCase, config: HydrodynamicConfig, *, fluid: FluidProperties = FluidProperties(),
                 endothelium: EndothelialControlVolume = EndothelialControlVolume(), include_fields: bool = False):
    case.validate(); config.validate(); fluid.validate(); endothelium.validate()
    solver = WomersleySolver(config.radial_order)
    alpha = case.radius_m * np.sqrt(fluid.angular_frequency_rad_s / fluid.kinematic_viscosity_m2_s)
    velocity_scale = case.pressure_gradient_scale_pa_per_m * case.radius_m**2 / fluid.dynamic_viscosity_pa_s
    fields, residuals = _harmonic_fields_with_fluid(solver, case, config, fluid)
    time_cycle = np.arange(config.time_points, dtype=float) / config.time_points
    h = np.arange(1, 7, dtype=float); basis = np.exp(1j * 2.0 * np.pi * np.outer(h, time_cycle))
    uz_full = np.real(fields["uz"] @ basis); ut_full = np.real(fields["ut"] @ basis)
    oz_full = np.real(fields["oz"] @ basis); ot_full = np.real(fields["ot"] @ basis)
    lamb_full = ut_full * oz_full - uz_full * ot_full
    d_uz_full = np.real((solver.D @ fields["uz"]) @ basis); d_ut_full = np.real((solver.D @ fields["ut"]) @ basis)
    kinetic_gradient = uz_full * d_uz_full + ut_full * d_ut_full
    radial_convective = np.zeros_like(ut_full); radial_convective[1:, :] = -(ut_full[1:, :] ** 2) / solver.r[1:, None]
    closure = kinetic_gradient - lamb_full - radial_convective
    closure_relative_error = float(np.max(np.abs(closure[1:, :])) / max(float(np.max(np.abs(kinetic_gradient))), 1e-30))
    epsilon = endothelium.thickness_m / case.radius_m
    near_wall_r = np.linspace(1.0 - epsilon, 1.0, config.near_wall_nodes)
    near = {key: _interpolate_columns(solver.r, value, near_wall_r) for key, value in fields.items()}
    uz = np.real(near["uz"] @ basis); ut = np.real(near["ut"] @ basis)
    oz = np.real(near["oz"] @ basis); ot = np.real(near["ot"] @ basis)
    lamb = ut * oz - uz * ot
    force_signed_n = endothelium.area_m2 * fluid.density_kg_m3 * velocity_scale**2 * np.trapezoid(lamb, near_wall_r, axis=0)
    force_exposure_n = endothelium.area_m2 * fluid.density_kg_m3 * velocity_scale**2 * np.trapezoid(np.abs(lamb), near_wall_r, axis=0)
    d_uz_h = solver.D @ fields["uz"]; d_ut_h = solver.D @ fields["ut"]
    shear_h = (d_uz_h[-1, :] + config.beta * (d_ut_h[-1, :] - fields["ut"][-1, :])) * (fluid.dynamic_viscosity_pa_s * velocity_scale / case.radius_m)
    wall_shear_pa = np.real(shear_h @ basis)
    result = {
        "artery_id": case.artery_id, "artery_name": case.name, "alpha": float(alpha),
        "alpha_reference": float(case.womersley_alpha_reference), "alpha_abs_error": float(abs(alpha-case.womersley_alpha_reference)),
        "velocity_scale_m_s": float(velocity_scale), "time_cycle": time_cycle,
        "time_s": time_cycle / fluid.fundamental_frequency_hz,
        "force_signed_n": force_signed_n, "force_exposure_n": force_exposure_n,
        "wall_shear_pa": wall_shear_pa, "max_normalized_backward_residual": float(max(residuals)),
        "differentiation_polynomial_error": solver.derivative_polynomial_error(),
        "gromeka_lamb_closure_relative_error": closure_relative_error, "config": asdict(config),
    }
    if include_fields:
        result.update({"near_wall_r_star": near_wall_r, "u_z_m_s": uz*velocity_scale, "u_theta_m_s": ut*velocity_scale,
                       "omega_z_s_inv": oz*velocity_scale/case.radius_m, "omega_theta_s_inv": ot*velocity_scale/case.radius_m,
                       "lamb_r_m_s2": lamb*velocity_scale**2/case.radius_m})
    return result

def compute_decomposition(case: ArteryCase, config: HydrodynamicConfig, *, fluid: FluidProperties = FluidProperties(),
                          endothelium: EndothelialControlVolume = EndothelialControlVolume()):
    anisotropic = compute_case(case, config, fluid=fluid, endothelium=endothelium)
    isotropic = compute_case(case, HydrodynamicConfig(config.radial_order, config.time_points, config.near_wall_nodes, 0.0, 0.0, 1.0),
                             fluid=fluid, endothelium=endothelium)
    return {
        "artery_id": case.artery_id, "artery_name": case.name,
        "time_cycle": anisotropic["time_cycle"], "time_s": anisotropic["time_s"],
        "wss_anisotropic_pa": anisotropic["wall_shear_pa"], "wss_isotropic_pa": isotropic["wall_shear_pa"],
        "force_signed_anisotropic_n": anisotropic["force_signed_n"], "force_signed_isotropic_n": isotropic["force_signed_n"],
        "force_signed_anisotropy_increment_n": np.asarray(anisotropic["force_signed_n"])-np.asarray(isotropic["force_signed_n"]),
        "force_exposure_anisotropic_n": anisotropic["force_exposure_n"], "force_exposure_isotropic_n": isotropic["force_exposure_n"],
        "force_exposure_anisotropy_increment_n": np.asarray(anisotropic["force_exposure_n"])-np.asarray(isotropic["force_exposure_n"]),
        "anisotropic_diagnostics": {k: anisotropic[k] for k in ("alpha","alpha_reference","alpha_abs_error","velocity_scale_m_s","max_normalized_backward_residual","differentiation_polynomial_error","gromeka_lamb_closure_relative_error")},
    }

def retain_harmonics(signal: np.ndarray, max_harmonic: int) -> np.ndarray:
    values = np.asarray(signal, dtype=float); coeff = np.fft.rfft(values); coeff[max_harmonic+1:] = 0.0
    return np.fft.irfft(coeff, n=values.size)
