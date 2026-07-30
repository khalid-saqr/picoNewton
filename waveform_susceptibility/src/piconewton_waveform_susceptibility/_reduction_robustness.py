from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES

from .core import (
    AnalysisConfig,
    exact_excess_kernel,
    near_wall_basis,
    relative_l2,
    second_order_kernel,
    unit_perturbation_response,
)

_EPS = 1e-30


def _fit_scale(records: Sequence[dict[str, Any]]) -> np.ndarray:
    design = np.asarray(
        [
            [1.0, np.log(record["alpha"]), np.log(record["eta"])]
            for record in records
        ]
    )
    response = np.log([record["kernel_norm"] for record in records])
    return np.linalg.lstsq(design, response, rcond=None)[0]


def _predict_scale(parameters: Sequence[float], alpha: float, eta: float) -> float:
    log_prefactor, alpha_exponent, eta_exponent = np.asarray(parameters, dtype=float)
    return float(np.exp(log_prefactor) * alpha**alpha_exponent * eta**eta_exponent)


def _rank_one(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    left, singular_values, right = np.linalg.svd(kernel, full_matrices=False)
    reduced = singular_values[0] * np.outer(left[:, 0], right[0, :])
    retained = float(singular_values[0] ** 2 / np.sum(singular_values**2))
    return reduced, singular_values, retained


def constitutive_robustness(
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    epsilon = config.validation_epsilon
    paths = (
        ("reciprocal", 1.0, 1.0, 1.0),
        ("beta_low", 0.5, 1.0, 1.0),
        ("gamma_low", 1.0, 0.5, 1.0),
        ("gamma_only", 0.0, 1.0, 1.0),
        ("beta_only", 1.0, 0.0, 1.0),
        ("beta_high_gamma_low", 1.25, 0.75, 1.0),
        ("beta_low_gamma_high", 0.75, 1.25, 1.0),
        ("delta_low", 1.0, 1.0, 0.8),
        ("delta_high", 1.0, 1.0, 1.2),
    )
    rows: list[dict[str, Any]] = []
    for case in V2_ARTERY_CASES:
        perturbation = near_wall_basis(
            case, unit_perturbation_response(case, config), config
        )
        frequencies, reciprocal_kernel = second_order_kernel(case, perturbation)
        reciprocal_norm = np.linalg.norm(reciprocal_kernel)
        reciprocal_shape = reciprocal_kernel / max(reciprocal_norm, _EPS)
        for name, beta_factor, gamma_factor, delta in paths:
            exact_frequencies, exact_kernel, residual = exact_excess_kernel(
                case,
                beta_factor * epsilon,
                gamma_factor * epsilon,
                delta,
                config,
            )
            if not np.array_equal(frequencies, exact_frequencies):
                raise RuntimeError("frequency axes disagree")
            scaled = exact_kernel / epsilon**2
            scaled_norm = np.linalg.norm(scaled)
            shape = scaled / max(scaled_norm, _EPS)
            null_control = gamma_factor == 0.0
            rows.append(
                {
                    "artery_id": case.artery_id,
                    "artery_name": case.name,
                    "constitutive_path": name,
                    "beta_factor": beta_factor,
                    "gamma_factor": gamma_factor,
                    "delta": delta,
                    "null_control": null_control,
                    "scaled_kernel_norm": float(scaled_norm),
                    "relative_amplitude_to_reciprocal": float(
                        scaled_norm / max(reciprocal_norm, _EPS)
                    ),
                    "normalised_shape_relative_l2": (
                        0.0
                        if null_control and scaled_norm <= 1e-20
                        else relative_l2(shape, reciprocal_shape)
                    ),
                    "maximum_residual": residual,
                }
            )
    return pd.DataFrame(rows)
