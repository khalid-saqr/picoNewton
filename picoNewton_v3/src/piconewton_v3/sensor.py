"""Dimensionally closed two-state mechanosensor kinetics."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

from .types import BOLTZMANN_J_PER_K, ForceMode, SensorConfig, StressMode

def thermal_energy_j(temperature_k: float) -> float:
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    return BOLTZMANN_J_PER_K * temperature_k


def lamb_work(
    force_n: np.ndarray | float,
    coupling_length_m: float,
    temperature_k: float,
    *,
    mode: ForceMode = "signed",
    signed_sensitivity: float = 1.0,
) -> np.ndarray:
    force = np.asarray(force_n, dtype=float)
    if coupling_length_m < 0:
        raise ValueError("coupling_length_m must be non-negative")
    if mode == "signed":
        effective = signed_sensitivity * force
    elif mode == "magnitude":
        effective = np.abs(force)
    elif mode == "outward_only":
        effective = np.maximum(force, 0.0)
    elif mode == "inward_only":
        effective = np.maximum(-force, 0.0)
    else:
        raise ValueError(f"unknown force mode: {mode}")
    return effective * coupling_length_m / thermal_energy_j(temperature_k)


def wss_work(
    wall_shear_pa: np.ndarray | float,
    activation_volume_m3: float,
    temperature_k: float,
    *,
    mode: StressMode = "signed",
    signed_sensitivity: float = 1.0,
) -> np.ndarray:
    stress = np.asarray(wall_shear_pa, dtype=float)
    if activation_volume_m3 < 0:
        raise ValueError("activation_volume_m3 must be non-negative")
    if mode == "signed":
        effective = signed_sensitivity * stress
    elif mode == "magnitude":
        effective = np.abs(stress)
    elif mode == "positive_only":
        effective = np.maximum(stress, 0.0)
    elif mode == "negative_only":
        effective = np.maximum(-stress, 0.0)
    else:
        raise ValueError(f"unknown stress mode: {mode}")
    return effective * activation_volume_m3 / thermal_energy_j(temperature_k)


def transition_rates(
    work: np.ndarray | float, sensor: SensorConfig
) -> tuple[np.ndarray, np.ndarray]:
    sensor.validate()
    psi = np.asarray(work, dtype=float)
    k_plus = (
        sensor.basal_probability
        / sensor.relaxation_time_s
        * np.exp(sensor.transition_fraction * psi)
    )
    k_minus = (
        (1.0 - sensor.basal_probability)
        / sensor.relaxation_time_s
        * np.exp(-(1.0 - sensor.transition_fraction) * psi)
    )
    return k_plus, k_minus


def equilibrium_probability(work: np.ndarray | float, sensor: SensorConfig) -> np.ndarray:
    sensor.validate()
    return expit(logit(sensor.basal_probability) + np.asarray(work, dtype=float))


def periodic_sensor_solution(
    work_cycle: np.ndarray,
    frequency_hz: float,
    sensor: SensorConfig,
) -> tuple[np.ndarray, float]:
    """Exact periodic solution for piecewise-constant work over one cycle."""
    sensor.validate()
    work = np.asarray(work_cycle, dtype=float)
    if work.ndim != 1 or len(work) < 2:
        raise ValueError("work_cycle must be a one-dimensional cycle")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    dt = 1.0 / frequency_hz / len(work)

    A = 1.0
    B = 0.0
    for psi in work:
        kp, km = transition_rates(psi, sensor)
        rate = float(kp + km)
        p_inf = float(kp / rate)
        a = float(np.exp(-rate * dt))
        b = p_inf * (1.0 - a)
        A, B = a * A, a * B + b
    if abs(1.0 - A) < 1e-14:
        raise RuntimeError("periodic affine map is numerically singular")
    state0 = B / (1.0 - A)

    probability = np.empty_like(work)
    state = state0
    for i, psi in enumerate(work):
        probability[i] = state
        kp, km = transition_rates(psi, sensor)
        rate = float(kp + km)
        p_inf = float(kp / rate)
        state = p_inf + (state - p_inf) * np.exp(-rate * dt)
    residual = float(abs(state - state0))
    if probability.min() < -1e-12 or probability.max() > 1.0 + 1e-12:
        raise RuntimeError("sensor probability left [0,1]")
    return probability, residual


def signal_metrics(signal: np.ndarray) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("signal must be a one-dimensional series")
    coefficients = np.fft.rfft(x) / len(x)
    power = np.abs(coefficients) ** 2
    return {
        "mean": float(np.mean(x)),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
        "rms": float(np.sqrt(np.mean(x**2))),
        "dynamic_range": float(np.ptp(x)),
        "peak_phase_cycle": float(np.argmax(x) / len(x)),
        "high_harmonic_power_fraction": float(power[3:].sum() / max(power.sum(), 1e-30)),
    }


def rms_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
