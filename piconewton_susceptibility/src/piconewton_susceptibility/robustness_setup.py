from __future__ import annotations

from typing import Any

import numpy as np
from piconewton_v3 import V2_ARTERY_CASES

from .reduction_core import (
    Step8Config,
    fit_power_law,
    kernel_scale,
    truncated_kernel,
    universal_kernel,
    waveform_catalog,
)
from .robustness_core import (
    DEFAULT_PATHS,
    Step9Config,
    alpha_for_case,
    derive_general_hierarchy,
    hierarchy_kernel,
    native_eta,
    relative_l2,
)

_EPS = 1e-30

def conditions(case: Any, config: Step9Config) -> tuple[tuple[str, float], ...]:
    return (
        ("hydrodynamic", config.eta_reference),
        ("physiological", native_eta(case)),
    )


def waveforms() -> list[dict[str, Any]]:
    return waveform_catalog(
        Step8Config(profile="publication", random_seed=20260730, phase_scrambles=8)
    )


def fit_scale(records: list[dict[str, Any]]) -> np.ndarray:
    return fit_power_law(
        np.asarray([row["alpha"] for row in records]),
        np.asarray([row["eta"] for row in records]),
        np.asarray([row["scale"] for row in records]),
    )


def build_hierarchy_cache(config: Step9Config) -> tuple[dict[tuple[str, float, int], Any], float]:
    cache: dict[tuple[str, float, int], Any] = {}
    maximum_residual = 0.0
    for case in V2_ARTERY_CASES:
        for delta in sorted({path.delta for path in DEFAULT_PATHS}):
            basis = derive_general_hierarchy(case, config.radial_order, delta)
            cache[(case.artery_id, delta, config.radial_order)] = basis
            maximum_residual = max(maximum_residual, basis.max_residual)
    return cache, maximum_residual


def reconstruct_reciprocal(
    config: Step9Config,
    hierarchy_cache: dict[tuple[str, float, int], Any],
    frozen_kernel: np.ndarray,
    frozen_parameters: np.ndarray,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], np.ndarray]]:
    records: list[dict[str, Any]] = []
    kernels: dict[tuple[str, str, str], np.ndarray] = {}
    for case in V2_ARTERY_CASES:
        basis = hierarchy_cache[(case.artery_id, 1.0, config.radial_order)]
        for matrix_type, eta in conditions(case, config):
            kernel = hierarchy_kernel(basis, eta, config.quadrature_nodes, 1.0, 1.0)
            kernels[("reciprocal", case.artery_id, matrix_type)] = kernel
            records.append(
                {
                    "vessel_id": case.artery_id,
                    "matrix_type": matrix_type,
                    "alpha": alpha_for_case(case),
                    "eta": eta,
                    "scale": kernel_scale(kernel),
                    "kernel": kernel,
                }
            )
    parameters = fit_scale(records)
    universal = universal_kernel(row["kernel"] for row in records)
    rank_one, _, energy = truncated_kernel(universal, 1)
    continuity = {
        "prefactor_relative_error": abs(
            float(np.exp(parameters[0])) - float(np.exp(frozen_parameters[0]))
        )
        / max(abs(float(np.exp(frozen_parameters[0]))), _EPS),
        "alpha_exponent_absolute_error": abs(
            float(parameters[1]) - float(frozen_parameters[1])
        ),
        "eta_exponent_absolute_error": abs(
            float(parameters[2]) - float(frozen_parameters[2])
        ),
        "selected_kernel_relative_l2": relative_l2(rank_one, frozen_kernel),
        "reconstructed_rank_one_energy": energy,
    }
    return continuity, kernels


