from __future__ import annotations

import numpy as np

from piconewton_susceptibility.susceptibility_core import (
    ExactNativeEvaluator,
    Step6Config,
    critical_epsilon_second_order,
    parseval_rms,
    scale_invariance_error,
    second_order_native,
    susceptibility_metrics,
)
from piconewton_v3 import V2_ARTERY_CASES


def _config() -> Step6Config:
    return Step6Config(
        profile="quick",
        radial_order=40,
        time_points=256,
        quadrature_nodes=64,
        validation_epsilons=(0.04, 0.08),
        inversion_verification_epsilons=(0.04,),
    )


def test_dimensionless_susceptibility_and_parseval_close() -> None:
    case = V2_ARTERY_CASES[0]
    result = second_order_native(case, _config())
    metrics = susceptibility_metrics(
        result["waveform_n_per_epsilon2"], result["force_scale_n"]
    )
    spectral = parseval_rms(result["spectrum"] / result["force_scale_n"])
    assert abs(spectral - metrics["phi_rms"]) / metrics["phi_rms"] < 1e-12
    assert metrics["phi_peak_abs"] >= metrics["phi_rms"] > 0.0
    assert metrics["outward_duty"] + metrics["inward_duty"] == 1.0


def test_pressure_scale_separates_from_susceptibility() -> None:
    errors = scale_invariance_error(V2_ARTERY_CASES[2], _config(), 2.0)
    assert errors["waveform_relative_l2"] < 1e-12
    assert errors["spectrum_relative_l2"] < 1e-12
    assert errors["force_scale_ratio_error"] < 1e-12


def test_inverse_predictor_and_full_crossing_recover_known_target() -> None:
    case = V2_ARTERY_CASES[0]
    config = _config()
    second = second_order_native(case, config)
    coefficient = float(
        np.sqrt(np.mean(np.asarray(second["waveform_n_per_epsilon2"]) ** 2))
    )
    evaluator = ExactNativeEvaluator(case, config)
    truth = 0.04
    target = evaluator.metric(truth, "rms")
    estimate = critical_epsilon_second_order(target, coefficient)
    status, crossing, _ = evaluator.refine_crossing(target, "rms", 0.08)
    assert abs(estimate - truth) / truth < 0.005
    assert status == "full_model_crossing_found"
    assert crossing is not None and abs(crossing - truth) < 2e-7


def test_formal_critical_estimate_is_positive() -> None:
    assert critical_epsilon_second_order(1e-12, 2e-11) == np.sqrt(0.05)
