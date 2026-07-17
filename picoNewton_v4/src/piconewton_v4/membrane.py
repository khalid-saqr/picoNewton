"""Reduced-order standard-linear-solid membrane--cortex interface.

The interface maps either wall shear stress or the signed endothelial
control-volume force to a common equivalent stress, then propagates that
stress through a passive standard-linear-solid (SLS) model. It does not
couple to Piezo1 and does not interpret the Lamb-force exposure magnitude as
signed mechanical loading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SLSParameters:
    """Standard-linear-solid parameters in SI units.

    The model is a relaxed spring ``E_inf`` in parallel with a Maxwell branch
    containing spring ``E_1 = E_0 - E_inf`` and dashpot ``eta = E_1*tau_sigma``.
    ``E_0`` is the instantaneous modulus and ``E_inf`` the relaxed modulus.
    """

    instantaneous_modulus_pa: float = 2500.0
    relaxed_modulus_pa: float = 1000.0
    stress_relaxation_time_s: float = 0.25
    thickness_m: float = 0.35e-6

    def validate(self, *, allow_elastic_limit: bool = True) -> None:
        values = np.asarray(
            [
                self.instantaneous_modulus_pa,
                self.relaxed_modulus_pa,
                self.stress_relaxation_time_s,
                self.thickness_m,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("SLS parameters must be finite")
        if self.instantaneous_modulus_pa <= 0 or self.relaxed_modulus_pa <= 0:
            raise ValueError("moduli must be positive")
        if self.instantaneous_modulus_pa < self.relaxed_modulus_pa:
            raise ValueError("instantaneous modulus must be >= relaxed modulus")
        if not allow_elastic_limit and self.instantaneous_modulus_pa == self.relaxed_modulus_pa:
            raise ValueError("primary SLS domain requires a nonzero Maxwell modulus")
        if self.stress_relaxation_time_s <= 0:
            raise ValueError("stress relaxation time must be positive")
        if self.thickness_m <= 0:
            raise ValueError("thickness must be positive")

    @property
    def maxwell_modulus_pa(self) -> float:
        return self.instantaneous_modulus_pa - self.relaxed_modulus_pa

    @property
    def viscosity_pa_s(self) -> float:
        return self.maxwell_modulus_pa * self.stress_relaxation_time_s

    @property
    def creep_time_s(self) -> float:
        return (
            self.stress_relaxation_time_s
            * self.instantaneous_modulus_pa
            / self.relaxed_modulus_pa
        )

    @property
    def is_elastic_limit(self) -> bool:
        return bool(np.isclose(self.maxwell_modulus_pa, 0.0, rtol=0.0, atol=1e-15))


@dataclass(frozen=True)
class MembraneAdmissibleDomain:
    """Source-backed engineering envelope used by the primary Step 6 analysis."""

    relaxed_modulus_pa: tuple[float, float] = (500.0, 4000.0)
    instantaneous_modulus_pa: tuple[float, float] = (1000.0, 5000.0)
    stress_relaxation_time_s: tuple[float, float] = (0.10, 0.50)
    thickness_m: tuple[float, float] = (0.20e-6, 0.50e-6)
    viscosity_pa_s: tuple[float, float] = (200.0, 1100.0)

    def contains(self, params: SLSParameters, *, allow_elastic_limit: bool = False) -> bool:
        try:
            params.validate(allow_elastic_limit=allow_elastic_limit)
        except ValueError:
            return False
        if not (self.relaxed_modulus_pa[0] <= params.relaxed_modulus_pa <= self.relaxed_modulus_pa[1]):
            return False
        if not (
            self.instantaneous_modulus_pa[0]
            <= params.instantaneous_modulus_pa
            <= self.instantaneous_modulus_pa[1]
        ):
            return False
        if not (
            self.stress_relaxation_time_s[0]
            <= params.stress_relaxation_time_s
            <= self.stress_relaxation_time_s[1]
        ):
            return False
        if not (self.thickness_m[0] <= params.thickness_m <= self.thickness_m[1]):
            return False
        if params.is_elastic_limit:
            return allow_elastic_limit
        return self.viscosity_pa_s[0] <= params.viscosity_pa_s <= self.viscosity_pa_s[1]

    def as_rows(self) -> list[dict[str, object]]:
        rows = []
        units = {
            "relaxed_modulus_pa": "Pa",
            "instantaneous_modulus_pa": "Pa",
            "stress_relaxation_time_s": "s",
            "thickness_m": "m",
            "viscosity_pa_s": "Pa s",
        }
        for name in units:
            low, high = getattr(self, name)
            rows.append({"parameter": name, "lower": low, "upper": high, "unit": units[name]})
        return rows


@dataclass(frozen=True)
class WSSLoadMap:
    """Map signed tangential WSS [Pa] to an equivalent scalar stress [Pa]."""

    transfer_fraction: float = 1.0

    def validate(self) -> None:
        if not np.isfinite(self.transfer_fraction) or not 0.0 <= self.transfer_fraction <= 1.0:
            raise ValueError("WSS transfer fraction must lie in [0, 1]")

    def map(self, wall_shear_stress_pa: np.ndarray) -> np.ndarray:
        self.validate()
        values = np.asarray(wall_shear_stress_pa, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("WSS input must be finite")
        return self.transfer_fraction * values


@dataclass(frozen=True)
class ForceLoadMap:
    """Map signed force [N] to equivalent scalar stress [Pa] using an area."""

    effective_area_m2: float = 100e-12
    transfer_fraction: float = 1.0

    def validate(self) -> None:
        if not np.isfinite(self.effective_area_m2) or self.effective_area_m2 <= 0:
            raise ValueError("effective area must be positive")
        if not np.isfinite(self.transfer_fraction) or not 0.0 <= self.transfer_fraction <= 1.0:
            raise ValueError("force transfer fraction must lie in [0, 1]")

    def map(self, signed_force_n: np.ndarray) -> np.ndarray:
        self.validate()
        values = np.asarray(signed_force_n, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("force input must be finite")
        return self.transfer_fraction * values / self.effective_area_m2


def complex_compliance(omega_rad_s: np.ndarray | float, params: SLSParameters) -> np.ndarray:
    """Complex creep compliance J*(omega) [Pa^-1] for exp(i omega t)."""

    params.validate()
    omega = np.asarray(omega_rad_s, dtype=float)
    if params.is_elastic_limit:
        return np.full_like(omega, 1.0 / params.instantaneous_modulus_pa, dtype=complex)
    numerator = 1.0 + 1j * omega * params.stress_relaxation_time_s
    denominator = (
        params.relaxed_modulus_pa
        + 1j
        * omega
        * params.stress_relaxation_time_s
        * params.instantaneous_modulus_pa
    )
    return numerator / denominator


def complex_modulus(omega_rad_s: np.ndarray | float, params: SLSParameters) -> np.ndarray:
    """Complex modulus E*(omega) [Pa]."""

    return 1.0 / complex_compliance(omega_rad_s, params)


def loss_modulus_pa(omega_rad_s: np.ndarray | float, params: SLSParameters) -> np.ndarray:
    """Nonnegative loss modulus for a passive SLS under exp(i omega t)."""

    return np.imag(complex_modulus(omega_rad_s, params))


def periodic_response(
    equivalent_stress_pa: np.ndarray,
    *,
    dt_s: float,
    params: SLSParameters,
) -> dict[str, np.ndarray]:
    """Return the periodic steady-state SLS response to a sampled stress cycle."""

    params.validate()
    stress = np.asarray(equivalent_stress_pa, dtype=float)
    if stress.ndim != 1 or stress.size < 8:
        raise ValueError("stress must be a one-dimensional periodic cycle with at least 8 points")
    if not np.all(np.isfinite(stress)):
        raise ValueError("stress must be finite")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be positive")

    omega = 2.0 * np.pi * np.fft.rfftfreq(stress.size, d=dt_s)
    strain_hat = np.fft.rfft(stress) * complex_compliance(omega, params)
    strain = np.fft.irfft(strain_hat, n=stress.size)
    maxwell_stress = stress - params.relaxed_modulus_pa * strain
    relaxed_branch_tension = params.thickness_m * params.relaxed_modulus_pa * strain
    maxwell_branch_tension = params.thickness_m * maxwell_stress
    total_applied_tension = params.thickness_m * stress

    if params.is_elastic_limit:
        dissipation_density = np.zeros_like(stress)
        stored_energy_density = 0.5 * params.relaxed_modulus_pa * strain**2
    else:
        dissipation_density = maxwell_stress**2 / params.viscosity_pa_s
        stored_energy_density = (
            0.5 * params.relaxed_modulus_pa * strain**2
            + 0.5 * maxwell_stress**2 / params.maxwell_modulus_pa
        )

    return {
        "equivalent_stress_pa": stress,
        "equivalent_areal_strain": strain,
        "relaxed_branch_tension_n_m": relaxed_branch_tension,
        "maxwell_branch_tension_n_m": maxwell_branch_tension,
        "total_applied_tension_n_m": total_applied_tension,
        "stored_energy_density_j_m3": stored_energy_density,
        "dissipation_density_w_m3": dissipation_density,
    }


def step_creep_response(
    time_s: np.ndarray,
    *,
    stress_step_pa: float,
    params: SLSParameters,
) -> np.ndarray:
    """Analytical strain after a stress step at t=0."""

    params.validate()
    time = np.asarray(time_s, dtype=float)
    if np.any(time < 0) or not np.all(np.isfinite(time)):
        raise ValueError("step-response time must be finite and nonnegative")
    if params.is_elastic_limit:
        return np.full_like(time, stress_step_pa / params.instantaneous_modulus_pa)
    initial = stress_step_pa / params.instantaneous_modulus_pa
    relaxed = stress_step_pa / params.relaxed_modulus_pa
    return relaxed + (initial - relaxed) * np.exp(-time / params.creep_time_s)


def harmonic_energy_balance(
    *,
    stress_amplitude_pa: float,
    frequency_hz: float,
    params: SLSParameters,
) -> Mapping[str, float]:
    """Analytical mean input power and dashpot dissipation for one sinusoid."""

    params.validate()
    if stress_amplitude_pa < 0 or frequency_hz <= 0:
        raise ValueError("amplitude must be nonnegative and frequency positive")
    omega = 2.0 * np.pi * frequency_hz
    compliance = complex(complex_compliance(omega, params))
    mean_input = -0.5 * omega * stress_amplitude_pa**2 * compliance.imag
    if params.is_elastic_limit:
        mean_dissipation = 0.0
    else:
        maxwell_amplitude = abs(1.0 - params.relaxed_modulus_pa * compliance) * stress_amplitude_pa
        mean_dissipation = 0.5 * maxwell_amplitude**2 / params.viscosity_pa_s
    relative_error = abs(mean_input - mean_dissipation) / max(abs(mean_input), abs(mean_dissipation), 1e-30)
    return {
        "mean_input_power_density_w_m3": float(mean_input),
        "mean_dissipation_density_w_m3": float(mean_dissipation),
        "relative_error": float(relative_error),
    }


def validate_passivity(params: SLSParameters, *, max_frequency_hz: float = 100.0) -> dict[str, float | bool]:
    """Check parameter inequalities, loss modulus and harmonic energy closure."""

    params.validate()
    frequencies = np.geomspace(1e-4, max_frequency_hz, 512)
    losses = loss_modulus_pa(2.0 * np.pi * frequencies, params)
    energy = harmonic_energy_balance(
        stress_amplitude_pa=1.0,
        frequency_hz=1.2,
        params=params,
    )
    minimum_loss = float(np.min(losses))
    passed = bool(minimum_loss >= -1e-12 and energy["relative_error"] < 1e-12)
    return {
        "minimum_loss_modulus_pa": minimum_loss,
        "energy_balance_relative_error": float(energy["relative_error"]),
        "passed": passed,
    }
