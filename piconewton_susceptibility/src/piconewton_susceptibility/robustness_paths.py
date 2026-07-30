from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES

from .reduction_core import (
    kernel_scale,
    predict_power_law,
    susceptibility_from_kernel,
    truncated_kernel,
    universal_kernel,
)
from .robustness_core import DEFAULT_PATHS, Step9Config, alpha_for_case, hierarchy_kernel, normalised_kernel_error
from .robustness_setup import conditions, fit_scale, waveforms

_EPS = 1e-30

def evaluate_constitutive_paths(
    config: Step9Config,
    hierarchy_cache: dict[tuple[str, float, int], Any],
    frozen_kernel: np.ndarray,
    frozen_parameters: np.ndarray,
    kernels: dict[tuple[str, str, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    catalogue = waveforms()
    path_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {
        "frozen_selected_kernel": frozen_kernel,
        "frozen_scale_parameters": frozen_parameters,
    }
    for path in DEFAULT_PATHS:
        records: list[dict[str, Any]] = []
        for case in V2_ARTERY_CASES:
            basis = hierarchy_cache[(case.artery_id, path.delta, config.radial_order)]
            for matrix_type, eta in conditions(case, config):
                kernel = hierarchy_kernel(
                    basis,
                    eta,
                    config.quadrature_nodes,
                    path.beta_ratio,
                    path.gamma_ratio,
                )
                kernels[(path.path_id, case.artery_id, matrix_type)] = kernel
                arrays[f"{path.path_id}__{case.artery_id}__{matrix_type}"] = kernel
                scale = kernel_scale(kernel)
                reciprocal = kernels[("reciprocal", case.artery_id, matrix_type)]
                records.append(
                    {
                        "vessel_id": case.artery_id,
                        "matrix_type": matrix_type,
                        "alpha": alpha_for_case(case),
                        "eta": eta,
                        "scale": scale,
                        "kernel": kernel,
                    }
                )
                scale_rows.append(
                    {
                        "path": path.path_id,
                        "vessel_id": case.artery_id,
                        "matrix_type": matrix_type,
                        "scale": scale,
                        "scale_ratio_to_reciprocal": scale
                        / max(kernel_scale(reciprocal), _EPS),
                        "normalised_kernel_error_to_reciprocal": (
                            normalised_kernel_error(kernel, reciprocal)
                            if scale > _EPS
                            else np.nan
                        ),
                    }
                )
        if path.path_id == "beta_only":
            path_rows.append(
                {
                    "path": path.path_id,
                    "beta_ratio": path.beta_ratio,
                    "gamma_ratio": path.gamma_ratio,
                    "delta": path.delta,
                    "prefactor_diagnostic": 0.0,
                    "alpha_exponent_diagnostic": np.nan,
                    "eta_exponent_diagnostic": np.nan,
                    "rank_one_energy": np.nan,
                }
            )
            continue
        diagnostic = fit_scale(records)
        path_universal = universal_kernel(row["kernel"] for row in records)
        _, _, path_energy = truncated_kernel(path_universal, 1)
        path_rows.append(
            {
                "path": path.path_id,
                "beta_ratio": path.beta_ratio,
                "gamma_ratio": path.gamma_ratio,
                "delta": path.delta,
                "prefactor_diagnostic": float(np.exp(diagnostic[0])),
                "alpha_exponent_diagnostic": float(diagnostic[1]),
                "eta_exponent_diagnostic": float(diagnostic[2]),
                "alpha_exponent_drift": float(diagnostic[1] - frozen_parameters[1]),
                "eta_exponent_drift": float(diagnostic[2] - frozen_parameters[2]),
                "rank_one_energy": path_energy,
            }
        )
        for record in records:
            frozen_scale = predict_power_law(
                frozen_parameters, record["alpha"], record["eta"]
            )
            for waveform in catalogue:
                coefficients = waveform["coefficients"]
                exact = susceptibility_from_kernel(record["kernel"], coefficients)
                shape_prediction = record["scale"] * susceptibility_from_kernel(
                    frozen_kernel, coefficients
                )
                frozen_prediction = frozen_scale * susceptibility_from_kernel(
                    frozen_kernel, coefficients
                )
                prediction_rows.append(
                    {
                        "path": path.path_id,
                        "vessel_id": record["vessel_id"],
                        "matrix_type": record["matrix_type"],
                        "waveform_id": waveform["waveform_id"],
                        "family": waveform["family"],
                        "exact_phi_rms": exact,
                        "shape_prediction": shape_prediction,
                        "shape_relative_error": abs(shape_prediction - exact)
                        / max(exact, _EPS),
                        "frozen_reciprocal_prediction": frozen_prediction,
                        "frozen_amplitude_relative_error": abs(
                            frozen_prediction - exact
                        )
                        / max(exact, _EPS),
                    }
                )
    return (
        pd.DataFrame(path_rows),
        pd.DataFrame(scale_rows),
        pd.DataFrame(prediction_rows),
        arrays,
    )


