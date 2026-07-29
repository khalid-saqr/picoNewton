from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from piconewton_v3 import EndothelialControlVolume, FluidProperties
from scipy.optimize import brentq

from .kernel_core import (
    Step5Config,
    canonical_coefficients,
    evaluate_kernel,
    force_scale,
    full_force_kernel,
    near_wall_basis,
    reconstruct_spectrum,
    relative_l2,
    second_order_force_kernel,
    unit_full_response,
    unit_perturbation_response,
)

_EPS = 1e-30


@dataclass(frozen=True)
class Step6Config:
    profile: str = "publication"
    radial_order: int = 150
    time_points: int = 2048
    quadrature_nodes: int = 256
    validation_epsilons: tuple[float, ...] = (0.04, 0.08, 0.10)
    inversion_verification_epsilons: tuple[float, ...] = (0.04, 0.08)
    force_benchmarks_pn: tuple[float, ...] = (1.0, 10.0)
    pressure_scale_factors: tuple[float, ...] = (0.5, 2.0)
    closure_tolerance: float = 1e-11
    cross_environment_exact_tolerance: float = 1e-8
    exact_validation_relative_limit: float = 0.01
    inverse_estimate_relative_limit: float = 0.005
    inverse_root_absolute_tolerance: float = 1e-7

    def validate(self) -> None:
        if self.profile not in {"quick", "publication"}:
            raise ValueError("profile must be quick or publication")
        if self.radial_order < 30 or self.time_points < 64 or self.quadrature_nodes < 8:
            raise ValueError("invalid numerical resolution")
        for values, name in (
            (self.validation_epsilons, "validation_epsilons"),
            (self.inversion_verification_epsilons, "inversion_verification_epsilons"),
        ):
            if not values or any(not 0.0 < value <= 0.1 for value in values):
                raise ValueError(f"{name} must lie in (0,0.1]")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
        if tuple(self.force_benchmarks_pn) != (1.0, 10.0):
            raise ValueError("publication force benchmarks are frozen at 1 and 10 pN")
        if any(value <= 0.0 for value in self.pressure_scale_factors):
            raise ValueError("pressure scale factors must be positive")
        if not 0.0 < self.closure_tolerance < 1.0:
            raise ValueError("closure_tolerance must lie in (0,1)")
        if not 0.0 < self.cross_environment_exact_tolerance < 1e-4:
            raise ValueError("cross_environment_exact_tolerance is invalid")
        if not 0.0 < self.exact_validation_relative_limit < 1.0:
            raise ValueError("exact_validation_relative_limit must lie in (0,1)")
        if not 0.0 < self.inverse_estimate_relative_limit < 1.0:
            raise ValueError("inverse_estimate_relative_limit must lie in (0,1)")
        if not 0.0 < self.inverse_root_absolute_tolerance < 1e-2:
            raise ValueError("inverse_root_absolute_tolerance is invalid")

    def step5_config(self) -> Step5Config:
        return Step5Config(
            profile=self.profile,
            radial_order=self.radial_order,
            time_points=self.time_points,
            quadrature_nodes=self.quadrature_nodes,
        )


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def alpha_for_case(case: Any) -> float:
    fluid = FluidProperties()
    return float(
        case.radius_m
        * np.sqrt(fluid.angular_frequency_rad_s / fluid.kinematic_viscosity_m2_s)
    )


def eta_for_case(case: Any) -> float:
    return float(EndothelialControlVolume().thickness_m / case.radius_m)


def susceptibility_metrics(waveform_n_per_epsilon2: np.ndarray, scale_n: float) -> dict[str, float]:
    waveform = np.asarray(waveform_n_per_epsilon2, dtype=float)
    phi = waveform / scale_n
    positive = np.maximum(phi, 0.0)
    negative = np.maximum(-phi, 0.0)
    return {
        "phi_rms": rms(phi),
        "phi_peak_abs": float(np.max(np.abs(phi))),
        "phi_positive_rms": rms(positive),
        "phi_negative_rms": rms(negative),
        "phi_mean": float(np.mean(phi)),
        "outward_duty": float(np.mean(phi > 0.0)),
        "inward_duty": float(np.mean(phi < 0.0)),
        "zero_duty": float(np.mean(phi == 0.0)),
    }


def parseval_rms(spectrum: np.ndarray) -> float:
    spectrum = np.asarray(spectrum, dtype=complex)
    return float(np.sqrt(np.sum(np.abs(spectrum) ** 2)))


def metric_value(waveform: np.ndarray, metric: str) -> float:
    waveform = np.asarray(waveform, dtype=float)
    if metric == "rms":
        return rms(waveform)
    if metric == "peak_abs":
        return float(np.max(np.abs(waveform)))
    raise ValueError(f"unsupported metric: {metric}")


def critical_epsilon_second_order(target_n: float, coefficient_n: float) -> float:
    if target_n <= 0.0:
        raise ValueError("target force must be positive")
    if coefficient_n <= 0.0:
        raise ValueError("susceptibility coefficient must be positive")
    return float(np.sqrt(target_n / coefficient_n))


class ExactNativeEvaluator:
    """Exact reciprocal full-model excess evaluator for one native artery waveform."""

    def __init__(self, case: Any, config: Step6Config):
        self.case = case
        self.config = config
        self.kernel_config = config.step5_config()
        self.frequencies, self.coefficients = canonical_coefficients(
            np.asarray(case.harmonic_coefficients, dtype=complex)
        )
        isotropic = unit_full_response(case, self.kernel_config, 0.0)
        _, isotropic_near_wall = near_wall_basis(case, isotropic, self.kernel_config)
        frequencies, self.isotropic_kernel = full_force_kernel(case, isotropic_near_wall)
        if not np.array_equal(frequencies, self.frequencies):
            raise RuntimeError("frequency axes disagree")
        self._cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def spectrum_and_waveform(self, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        epsilon = float(epsilon)
        if epsilon < 0.0 or epsilon > 0.1:
            raise ValueError("epsilon must lie in [0,0.1]")
        key = round(epsilon, 14)
        if key in self._cache:
            return self._cache[key]
        if epsilon == 0.0:
            output_frequencies = np.arange(-12, 13, dtype=int)
            spectrum = np.zeros(25, dtype=complex)
            waveform = np.zeros(self.config.time_points, dtype=float)
        else:
            anisotropic = unit_full_response(self.case, self.kernel_config, epsilon)
            _, anisotropic_near_wall = near_wall_basis(
                self.case, anisotropic, self.kernel_config
            )
            frequencies, anisotropic_kernel = full_force_kernel(
                self.case, anisotropic_near_wall
            )
            if not np.array_equal(frequencies, self.frequencies):
                raise RuntimeError("frequency axes disagree")
            excess_kernel = anisotropic_kernel - self.isotropic_kernel
            output_frequencies, spectrum, _ = evaluate_kernel(
                self.frequencies, excess_kernel, self.coefficients
            )
            waveform = np.real(
                reconstruct_spectrum(output_frequencies, spectrum, self.config.time_points)
            )
        self._cache[key] = (output_frequencies, spectrum, waveform)
        return self._cache[key]

    def metric(self, epsilon: float, metric: str) -> float:
        return metric_value(self.spectrum_and_waveform(epsilon)[2], metric)

    def monotonic_metric(self, metric: str, upper: float, points: int = 9) -> bool:
        values = np.array(
            [self.metric(value, metric) for value in np.linspace(0.0, upper, points)]
        )
        tolerance = max(np.max(values), _EPS) * 1e-10
        return bool(np.all(np.diff(values) >= -tolerance))

    def refine_crossing(
        self,
        target_n: float,
        metric: str,
        upper: float,
    ) -> tuple[str, float | None, float]:
        if target_n <= 0.0:
            raise ValueError("target force must be positive")
        if not 0.0 < upper <= 0.1:
            raise ValueError("upper validity limit must lie in (0,0.1]")
        maximum = self.metric(upper, metric)
        if maximum < target_n:
            return "unreachable_within_validated_domain", None, maximum
        if not self.monotonic_metric(metric, upper):
            return "nonmonotonic_within_validated_domain", None, maximum
        crossing = brentq(
            lambda value: self.metric(value, metric) - target_n,
            0.0,
            upper,
            xtol=self.config.inverse_root_absolute_tolerance,
            rtol=1e-12,
            maxiter=60,
        )
        return "full_model_crossing_found", float(crossing), maximum


def second_order_native(case: Any, config: Step6Config) -> dict[str, Any]:
    kernel_config = config.step5_config()
    perturbation = unit_perturbation_response(case, kernel_config)
    _, near_wall = near_wall_basis(case, perturbation, kernel_config)
    frequencies, kernel = second_order_force_kernel(case, near_wall)
    _, coefficients = canonical_coefficients(
        np.asarray(case.harmonic_coefficients, dtype=complex)
    )
    output_frequencies, spectrum, _ = evaluate_kernel(frequencies, kernel, coefficients)
    waveform = np.real(reconstruct_spectrum(output_frequencies, spectrum, config.time_points))
    return {
        "frequencies": frequencies,
        "kernel": kernel,
        "output_frequencies": output_frequencies,
        "spectrum": spectrum,
        "waveform_n_per_epsilon2": waveform,
        "force_scale_n": force_scale(case),
        "max_residual": perturbation.max_residual,
    }


def scale_invariance_error(case: Any, config: Step6Config, factor: float) -> dict[str, float]:
    baseline = second_order_native(case, config)
    scaled_case = replace(
        case,
        pressure_gradient_scale_pa_per_m=case.pressure_gradient_scale_pa_per_m * factor,
    )
    scaled = second_order_native(scaled_case, config)
    phi_baseline = baseline["waveform_n_per_epsilon2"] / baseline["force_scale_n"]
    phi_scaled = scaled["waveform_n_per_epsilon2"] / scaled["force_scale_n"]
    spectrum_baseline = baseline["spectrum"] / baseline["force_scale_n"]
    spectrum_scaled = scaled["spectrum"] / scaled["force_scale_n"]
    return {
        "waveform_relative_l2": relative_l2(phi_scaled, phi_baseline),
        "spectrum_relative_l2": relative_l2(spectrum_scaled, spectrum_baseline),
        "force_scale_ratio_error": abs(
            scaled["force_scale_n"] / baseline["force_scale_n"] - factor**2
        ),
    }
