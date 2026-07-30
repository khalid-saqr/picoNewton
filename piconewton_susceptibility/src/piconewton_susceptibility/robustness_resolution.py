from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES

from .reduction_core import susceptibility_from_kernel
from .robustness_core import Step9Config, derive_general_hierarchy, hierarchy_kernel, relative_l2
from .robustness_setup import conditions, waveforms

_EPS = 1e-30

def evaluate_resolution_robustness(
    config: Step9Config,
    kernels: dict[tuple[str, str, str], np.ndarray],
) -> tuple[pd.DataFrame, float]:
    catalogue = waveforms()
    representatives = (catalogue[0], catalogue[6], catalogue[11], catalogue[-1])
    rows: list[dict[str, Any]] = []
    maximum_residual = 0.0
    for radial_order, quadrature_nodes in config.resolution_pairs:
        for case in V2_ARTERY_CASES:
            basis = derive_general_hierarchy(case, radial_order, 1.0)
            maximum_residual = max(maximum_residual, basis.max_residual)
            for matrix_type, eta in conditions(case, config):
                kernel = hierarchy_kernel(basis, eta, quadrature_nodes, 1.0, 1.0)
                reference = kernels[("reciprocal", case.artery_id, matrix_type)]
                for waveform in representatives:
                    exact = susceptibility_from_kernel(
                        reference, waveform["coefficients"]
                    )
                    perturbed = susceptibility_from_kernel(
                        kernel, waveform["coefficients"]
                    )
                    rows.append(
                        {
                            "radial_order": radial_order,
                            "quadrature_nodes": quadrature_nodes,
                            "vessel_id": case.artery_id,
                            "matrix_type": matrix_type,
                            "waveform_id": waveform["waveform_id"],
                            "relative_change": abs(perturbed - exact)
                            / max(exact, _EPS),
                            "kernel_relative_change": relative_l2(kernel, reference),
                        }
                    )
    return pd.DataFrame(rows), maximum_residual
