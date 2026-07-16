"""Runtime numerical verification dashboard."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr

from .model import (
    ArteryCase, EndothelialControlVolume, FluidProperties, HydrodynamicConfig,
    SensorConfig, V2_ARTERY_CASES, compute_hydrodynamics, lamb_work,
    periodic_sensor_solution, rms_difference, signal_metrics, wss_work,
)
from .study_io import StudyStore
from .workflow_common import (
    DEFAULT_SEED, NUMERICAL_SENSOR_UNCERTAINTY, _phase_distance, _rms,
    _sensor_from_work, run_hydrodynamic_cases,
)

def runtime_verification_dashboard(
    cases: Sequence[ArteryCase],
    config: HydrodynamicConfig,
    hydrodynamics: dict[str, dict[str, Any]],
    control_table: pd.DataFrame,
    fluid: FluidProperties = FluidProperties(),
    endothelium: EndothelialControlVolume = EndothelialControlVolume(),
) -> pd.DataFrame:
    """Execute publication guards at the active numerical profile."""
    from .model import WomersleySolver, isotropic_validation

    rows: list[dict[str, Any]] = []
    validation = isotropic_validation(radial_order=config.radial_order)
    iso_error = max(float(item["linf_error"]) for item in validation)
    rows.append(
        {
            "test": "isotropic_analytic",
            "observed": iso_error,
            "threshold": 1e-8,
            "passed": iso_error < 1e-8,
        }
    )
    polynomial_error = WomersleySolver(config.radial_order, "verified").derivative_polynomial_error()
    rows.append(
        {
            "test": "differentiation_polynomial",
            "observed": polynomial_error,
            "threshold": 1e-10,
            "passed": polynomial_error < 1e-10,
        }
    )
    max_residual = max(
        float(item["max_normalized_backward_residual"]) for item in hydrodynamics.values()
    )
    rows.append(
        {
            "test": "normalized_backward_residual",
            "observed": max_residual,
            "threshold": 1e-13,
            "passed": max_residual < 1e-13,
        }
    )
    periodic_residual = float(control_table["periodic_residual"].max())
    rows.append(
        {
            "test": "sensor_periodic_closure",
            "observed": periodic_residual,
            "threshold": 1e-10,
            "passed": periodic_residual < 1e-10,
        }
    )

    # Six input harmonics can create nonlinear support only through h=12.
    aliasing = 0.0
    for item in hydrodynamics.values():
        force = np.asarray(item["force_signed_n"])
        power = np.abs(np.fft.rfft(force) / len(force)) ** 2
        harmonics = np.arange(len(power))
        aliasing = max(aliasing, float(power[harmonics > 12].sum() / max(power.sum(), 1e-30)))
    rows.append(
        {
            "test": "power_above_h12",
            "observed": aliasing,
            "threshold": 1e-12,
            "passed": aliasing < 1e-12,
        }
    )

    # Representative active-profile convergence guards.
    representative = cases[0]
    reference_time = HydrodynamicConfig(
        radial_order=config.radial_order,
        time_points=config.time_points * 2,
        quadrature_nodes=config.quadrature_nodes,
        beta=config.beta,
        gamma=config.gamma,
        delta=config.delta,
        mode="verified",
    )
    time_ref = compute_hydrodynamics(representative, reference_time, fluid, endothelium)
    base_force = np.asarray(hydrodynamics[representative.artery_id]["force_signed_n"])
    reference_force = np.asarray(time_ref["force_signed_n"])[::2]
    time_error = float(
        np.linalg.norm(base_force - reference_force)
        / max(np.linalg.norm(reference_force), 1e-30)
    )
    rows.append(
        {
            "test": "time_force_relative_l2",
            "observed": time_error,
            "threshold": 1e-4,
            "passed": time_error < 1e-4,
        }
    )

    reference_quad = HydrodynamicConfig(
        radial_order=config.radial_order,
        time_points=config.time_points,
        quadrature_nodes=config.quadrature_nodes * 2,
        beta=config.beta,
        gamma=config.gamma,
        delta=config.delta,
        mode="verified",
    )
    quad_ref = compute_hydrodynamics(representative, reference_quad, fluid, endothelium)
    quad_force = np.asarray(quad_ref["force_signed_n"])
    quad_error = float(
        np.linalg.norm(base_force - quad_force) / max(np.linalg.norm(quad_force), 1e-30)
    )
    rows.append(
        {
            "test": "near_wall_quadrature_relative_l2",
            "observed": quad_error,
            "threshold": 1e-4,
            "passed": quad_error < 1e-4,
        }
    )
    return pd.DataFrame(rows)
