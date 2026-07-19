"""Vector-resolved, spatially partitioned membrane-cortex interface.

Tangential WSS, signed transverse force, and nonnegative Lamb-force exposure
remain distinct observables. Tangential and normal loads pass through separate
passive fast/slow viscoelastic compliances. Positive and negative normal strain
are routed to different membrane domains, while the magnitude exposure drives
an explicitly nonnegative energy/tension pathway.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

MMHG_PA = 133.32

@dataclass(frozen=True)
class DirectionalSLS:
    instantaneous_modulus_pa: float
    relaxed_modulus_pa: float
    fast_time_s: float
    slow_time_s: float
    fast_fraction: float
    def validate(self, *, allow_elastic: bool = True) -> None:
        vals = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.all(np.isfinite(vals)):
            raise ValueError("directional SLS parameters must be finite")
        if self.instantaneous_modulus_pa <= 0 or self.relaxed_modulus_pa <= 0:
            raise ValueError("moduli must be positive")
        if self.instantaneous_modulus_pa < self.relaxed_modulus_pa:
            raise ValueError("instantaneous modulus must be >= relaxed modulus")
        if not allow_elastic and self.instantaneous_modulus_pa == self.relaxed_modulus_pa:
            raise ValueError("nonelastic branch required")
        if self.fast_time_s <= 0 or self.slow_time_s <= 0 or self.fast_time_s > self.slow_time_s:
            raise ValueError("require 0 < fast_time <= slow_time")
        if not 0.0 <= self.fast_fraction <= 1.0:
            raise ValueError("fast_fraction must lie in [0,1]")
    @property
    def is_elastic(self) -> bool:
        return bool(np.isclose(self.instantaneous_modulus_pa, self.relaxed_modulus_pa, atol=1e-15, rtol=0.0))

def _sls_compliance(omega: np.ndarray, p: DirectionalSLS, tau: float) -> np.ndarray:
    if p.is_elastic:
        return np.full_like(omega, 1.0/p.instantaneous_modulus_pa, dtype=complex)
    return (1.0 + 1j*omega*tau) / (p.relaxed_modulus_pa + 1j*omega*tau*p.instantaneous_modulus_pa)

def generalized_compliance(omega_rad_s: np.ndarray | float, p: DirectionalSLS) -> np.ndarray:
    p.validate(); omega = np.asarray(omega_rad_s, dtype=float)
    jf = _sls_compliance(omega, p, p.fast_time_s)
    js = _sls_compliance(omega, p, p.slow_time_s)
    return p.fast_fraction*jf + (1.0-p.fast_fraction)*js

def periodic_strain(stress_pa: np.ndarray, *, dt_s: float, p: DirectionalSLS) -> np.ndarray:
    p.validate(); stress = np.asarray(stress_pa, dtype=float)
    if stress.ndim != 1 or stress.size < 8 or not np.all(np.isfinite(stress)) or dt_s <= 0:
        raise ValueError("invalid periodic stress")
    omega = 2*np.pi*np.fft.rfftfreq(stress.size, d=dt_s)
    return np.fft.irfft(np.fft.rfft(stress)*generalized_compliance(omega, p), n=stress.size)

def validate_passivity(p: DirectionalSLS) -> dict[str, float | bool]:
    p.validate(); frequency = np.geomspace(1e-4, 100.0, 1024)
    compliance = generalized_compliance(2*np.pi*frequency, p)
    minimum_loss_compliance = float(np.min(-np.imag(compliance)))
    return {"minimum_minus_imag_compliance_pa_inv": minimum_loss_compliance,
            "passed": bool(minimum_loss_compliance >= -1e-14)}

@dataclass(frozen=True)
class VectorInterfaceParameters:
    tangential: DirectionalSLS = DirectionalSLS(2500.0, 1000.0, 0.010, 0.250, 0.25)
    normal: DirectionalSLS = DirectionalSLS(1800.0, 700.0, 0.005, 0.150, 0.40)
    signed_force_area_m2: float = 100e-12
    exposure_area_m2: float = 100e-12
    wss_transfer_fraction: float = 1.0
    signed_force_transfer_fraction: float = 1.0
    exposure_transfer_fraction: float = 1.0
    areal_modulus_n_m: float = 0.20
    baseline_apical_tension_n_m: float = 0.50e-3
    baseline_junctional_tension_n_m: float = 0.50e-3
    apical_curvature_radius_m: float = 1.0e-6
    junctional_curvature_radius_m: float = 1.0e-6
    tangential_to_apical_cross_fraction: float = 0.0
    exposure_to_junctional_cross_fraction: float = 0.0
    maximum_pressure_mmhg: float = 70.0
    gradmu_mv: float = -40.0
    apical_channel_fraction: float = 0.5
    def validate(self) -> None:
        self.tangential.validate(); self.normal.validate()
        positive = [self.signed_force_area_m2, self.exposure_area_m2, self.areal_modulus_n_m,
                    self.apical_curvature_radius_m, self.junctional_curvature_radius_m,
                    self.maximum_pressure_mmhg]
        if not np.all(np.isfinite(positive)) or np.min(positive) <= 0:
            raise ValueError("positive vector-interface parameters required")
        fractions = [self.wss_transfer_fraction, self.signed_force_transfer_fraction,
                     self.exposure_transfer_fraction, self.tangential_to_apical_cross_fraction,
                     self.exposure_to_junctional_cross_fraction, self.apical_channel_fraction]
        if not np.all(np.isfinite(fractions)) or any(x < 0 or x > 1 for x in fractions):
            raise ValueError("all transfer/population fractions must lie in [0,1]")
        if self.baseline_apical_tension_n_m < 0 or self.baseline_junctional_tension_n_m < 0:
            raise ValueError("baseline tensions must be nonnegative")

def vector_membrane_state(*, wall_shear_pa: np.ndarray, signed_force_n: np.ndarray,
                          force_exposure_n: np.ndarray, dt_s: float,
                          p: VectorInterfaceParameters = VectorInterfaceParameters()) -> dict[str, np.ndarray | float]:
    p.validate()
    wss = np.asarray(wall_shear_pa, dtype=float); signed = np.asarray(signed_force_n, dtype=float)
    exposure = np.asarray(force_exposure_n, dtype=float)
    if wss.shape != signed.shape or wss.shape != exposure.shape or wss.ndim != 1:
        raise ValueError("all load waveforms must share one-dimensional shape")
    if not np.all(np.isfinite(wss)) or not np.all(np.isfinite(signed)) or not np.all(np.isfinite(exposure)):
        raise ValueError("load waveforms must be finite")
    if np.min(exposure) < -1e-18:
        raise ValueError("magnitude exposure must be nonnegative")
    tangential_traction = p.wss_transfer_fraction*wss
    normal_signed_traction = p.signed_force_transfer_fraction*signed/p.signed_force_area_m2
    normal_exposure_traction = p.exposure_transfer_fraction*exposure/p.exposure_area_m2
    eps_t = periodic_strain(tangential_traction, dt_s=dt_s, p=p.tangential)
    eps_n = periodic_strain(normal_signed_traction, dt_s=dt_s, p=p.normal)
    eps_exp = periodic_strain(normal_exposure_traction, dt_s=dt_s, p=p.normal)
    eps_exp = np.maximum(eps_exp, 0.0)
    normal_positive = np.maximum(eps_n, 0.0); normal_negative = np.maximum(-eps_n, 0.0)
    tangential_magnitude = np.abs(eps_t)
    apical_increment = p.areal_modulus_n_m*(
        normal_positive + eps_exp + p.tangential_to_apical_cross_fraction*tangential_magnitude)
    junctional_increment = p.areal_modulus_n_m*(
        tangential_magnitude + normal_negative + p.exposure_to_junctional_cross_fraction*eps_exp)
    apical_tension = np.maximum(p.baseline_apical_tension_n_m + apical_increment, 0.0)
    junctional_tension = np.maximum(p.baseline_junctional_tension_n_m + junctional_increment, 0.0)
    p_apical_raw = 2*apical_tension/p.apical_curvature_radius_m/MMHG_PA
    p_junctional_raw = 2*junctional_tension/p.junctional_curvature_radius_m/MMHG_PA
    p_apical = np.clip(p_apical_raw, 0.0, p.maximum_pressure_mmhg)
    p_junctional = np.clip(p_junctional_raw, 0.0, p.maximum_pressure_mmhg)
    angle = np.arctan2(normal_signed_traction, tangential_traction)
    vector_magnitude = np.hypot(tangential_traction, normal_signed_traction)
    return {
        "tangential_traction_pa": tangential_traction,
        "normal_signed_traction_pa": normal_signed_traction,
        "normal_exposure_traction_pa": normal_exposure_traction,
        "tangential_strain": eps_t, "normal_signed_strain": eps_n,
        "normal_exposure_strain": eps_exp,
        "apical_tension_n_m": apical_tension, "junctional_tension_n_m": junctional_tension,
        "apical_pressure_mmhg": p_apical, "junctional_pressure_mmhg": p_junctional,
        "force_vector_angle_rad": angle, "force_vector_magnitude_pa": vector_magnitude,
        "apical_pressure_clipped_fraction": float(np.mean(p_apical_raw > p.maximum_pressure_mmhg)),
        "junctional_pressure_clipped_fraction": float(np.mean(p_junctional_raw > p.maximum_pressure_mmhg)),
        "exposure_used_as_signed_load": False,
    }

def elastic_limit(p: VectorInterfaceParameters) -> VectorInterfaceParameters:
    tangential = DirectionalSLS(p.tangential.instantaneous_modulus_pa, p.tangential.instantaneous_modulus_pa,
                                p.tangential.fast_time_s, p.tangential.slow_time_s, p.tangential.fast_fraction)
    normal = DirectionalSLS(p.normal.instantaneous_modulus_pa, p.normal.instantaneous_modulus_pa,
                            p.normal.fast_time_s, p.normal.slow_time_s, p.normal.fast_fraction)
    values = asdict(p); values["tangential"] = tangential; values["normal"] = normal
    return VectorInterfaceParameters(**values)
