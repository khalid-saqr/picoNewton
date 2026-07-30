from __future__ import annotations

from typing import Any

import pandas as pd

from .robustness_core import Step9Config
from .robustness_support import error_summary


def calculate_metrics_and_gates(
    config: Step9Config,
    continuity: dict[str, Any],
    path_frame: pd.DataFrame,
    scale_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    exact_frame: pd.DataFrame,
    eta_frame: pd.DataFrame,
    resolution_frame: pd.DataFrame,
    maximum_residual: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metric_rows = []
    for path_id, current in prediction_frame.groupby("path"):
        shape = error_summary(current["shape_relative_error"])
        amplitude = error_summary(current["frozen_amplitude_relative_error"])
        metric_rows.append(
            {
                "path": path_id,
                **{f"shape_{name}": value for name, value in shape.items()},
                **{
                    f"frozen_amplitude_{name}": value
                    for name, value in amplitude.items()
                },
            }
        )
    metric_frame = pd.DataFrame(metric_rows)
    finite_nonnull = exact_frame.query("path != 'beta_only'")
    finite = error_summary(finite_nonnull["kernel_relative_error"])
    beta_exact = float(exact_frame.query("path == 'beta_only'")["exact_scale"].max())
    nonreciprocal = prediction_frame.query("path != 'reciprocal'")
    shape = error_summary(nonreciprocal["shape_relative_error"])
    eta = error_summary(eta_frame["relative_error"])
    resolution = error_summary(resolution_frame["relative_change"])
    minimum_energy = float(
        path_frame.query("path != 'beta_only'")["rank_one_energy"].min()
    )
    beta_hierarchy = float(scale_frame.query("path == 'beta_only'")["scale"].max())
    amplitude_maximum = float(
        nonreciprocal["frozen_amplitude_relative_error"].max()
    )
    kernel_limit = 1e-8 if config.profile == "publication" else 1e-5
    gates: dict[str, Any] = {
        "step": 9,
        "profile": config.profile,
        "step8_gate_consumed": True,
        "frozen_law_not_refitted": True,
        "step8_prefactor_continuity": continuity["prefactor_relative_error"] <= 5e-4,
        "step8_alpha_exponent_continuity": continuity[
            "alpha_exponent_absolute_error"
        ]
        <= 5e-4,
        "step8_eta_exponent_continuity": continuity["eta_exponent_absolute_error"]
        <= 5e-4,
        "step8_kernel_continuity": continuity["selected_kernel_relative_l2"]
        <= kernel_limit,
        "finite_epsilon_maximum_error": finite["maximum"],
        "finite_epsilon_closure_passed": finite["maximum"]
        <= config.finite_epsilon_error_max,
        "beta_only_exact_maximum_scale": beta_exact,
        "beta_only_exact_null_passed": beta_exact <= 1e-12,
        "nonreciprocal_shape_median_error": shape["median"],
        "nonreciprocal_shape_maximum_error": shape["maximum"],
        "nonreciprocal_shape_passed": shape["median"]
        <= config.shape_median_error_max
        and shape["maximum"] <= config.shape_maximum_error_max,
        "minimum_nonnull_rank_one_energy": minimum_energy,
        "rank_one_structure_passed": minimum_energy >= config.rank_one_energy_min,
        "eta_median_error": eta["median"],
        "eta_p90_error": eta["p90"],
        "eta_maximum_error": eta["maximum"],
        "eta_robustness_passed": eta["median"] <= config.eta_median_error_max
        and eta["p90"] <= config.eta_p90_error_max
        and eta["maximum"] <= config.eta_maximum_error_max,
        "resolution_maximum_change": resolution["maximum"],
        "resolution_passed": resolution["maximum"] <= config.resolution_change_max,
        "beta_only_maximum_scale": beta_hierarchy,
        "beta_only_null_passed": beta_hierarchy <= 1e-12,
        "maximum_solver_backward_residual": maximum_residual,
        "solver_residual_passed": maximum_residual <= config.residual_max,
        "uncorrected_nonreciprocal_amplitude_maximum_error": amplitude_maximum,
        "uncorrected_amplitude_universal": amplitude_maximum
        <= config.shape_maximum_error_max,
        "amplitude_claim_restricted_to_reciprocal": amplitude_maximum
        > config.shape_maximum_error_max,
        "constitutive_robustness_run": True,
        "biological_endpoint_model_run": False,
    }
    excluded = {
        "step",
        "profile",
        "finite_epsilon_maximum_error",
        "nonreciprocal_shape_median_error",
        "nonreciprocal_shape_maximum_error",
        "minimum_nonnull_rank_one_energy",
        "eta_median_error",
        "eta_p90_error",
        "eta_maximum_error",
        "resolution_maximum_change",
        "beta_only_exact_maximum_scale",
        "beta_only_maximum_scale",
        "maximum_solver_backward_residual",
        "uncorrected_nonreciprocal_amplitude_maximum_error",
        "uncorrected_amplitude_universal",
        "biological_endpoint_model_run",
    }
    gates["passed"] = all(
        bool(value) for name, value in gates.items() if name not in excluded
    )
    return metric_frame, gates
