from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES

from .robustness_core import (
    DEFAULT_PATHS,
    Step9Config,
    full_excess_kernel,
    kernel_scale,
    relative_l2,
    solve_full_response,
)
from .robustness_setup import conditions

def evaluate_exact_closure(
    config: Step9Config,
    kernels: dict[tuple[str, str, str], np.ndarray],
) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, Any]] = []
    maximum_residual = 0.0
    for case in V2_ARTERY_CASES:
        isotropic = solve_full_response(case, config.radial_order, 0.0, 0.0, 1.0)
        maximum_residual = max(maximum_residual, isotropic.max_residual)
        for path in DEFAULT_PATHS:
            anisotropic = solve_full_response(
                case,
                config.radial_order,
                path.beta_ratio * config.exact_epsilon,
                path.gamma_ratio * config.exact_epsilon,
                path.delta,
            )
            maximum_residual = max(maximum_residual, anisotropic.max_residual)
            for matrix_type, eta in conditions(case, config):
                exact = full_excess_kernel(
                    anisotropic, isotropic, eta, config.quadrature_nodes
                ) / config.exact_epsilon**2
                hierarchy = kernels[(path.path_id, case.artery_id, matrix_type)]
                rows.append(
                    {
                        "path": path.path_id,
                        "vessel_id": case.artery_id,
                        "matrix_type": matrix_type,
                        "epsilon": config.exact_epsilon,
                        "kernel_relative_error": relative_l2(hierarchy, exact),
                        "hierarchy_scale": kernel_scale(hierarchy),
                        "exact_scale": kernel_scale(exact),
                    }
                )
    return pd.DataFrame(rows), maximum_residual


