from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES

from .reduction_core import predict_power_law, susceptibility_from_kernel
from .robustness_core import Step9Config, alpha_for_case, hierarchy_kernel, native_eta
from .robustness_setup import waveforms

_EPS = 1e-30

def evaluate_eta_robustness(
    config: Step9Config,
    hierarchy_cache: dict[tuple[str, float, int], Any],
    frozen_kernel: np.ndarray,
    frozen_parameters: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in V2_ARTERY_CASES:
        basis = hierarchy_cache[(case.artery_id, 1.0, config.radial_order)]
        for multiplier in config.eta_multipliers:
            eta = native_eta(case) * multiplier
            kernel = hierarchy_kernel(basis, eta, config.quadrature_nodes, 1.0, 1.0)
            scale = predict_power_law(frozen_parameters, alpha_for_case(case), eta)
            for waveform in waveforms():
                exact = susceptibility_from_kernel(kernel, waveform["coefficients"])
                predicted = scale * susceptibility_from_kernel(
                    frozen_kernel, waveform["coefficients"]
                )
                rows.append(
                    {
                        "vessel_id": case.artery_id,
                        "eta_multiplier": multiplier,
                        "waveform_id": waveform["waveform_id"],
                        "family": waveform["family"],
                        "relative_error": abs(predicted - exact) / max(exact, _EPS),
                    }
                )
    return pd.DataFrame(rows)


