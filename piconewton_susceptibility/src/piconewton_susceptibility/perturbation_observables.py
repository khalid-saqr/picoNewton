from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from piconewton_v3 import EndothelialControlVolume, FluidProperties
from scipy.interpolate import BarycentricInterpolator

from .perturbation_core import HarmonicHierarchy, Step4Config


def interpolate_columns(
    radial_nodes: np.ndarray, values: np.ndarray, query: np.ndarray
) -> np.ndarray:
    return np.stack(
        [
            BarycentricInterpolator(radial_nodes, values[:, j])(query)
            for j in range(values.shape[1])
        ],
        axis=1,
    )


def real_fields(
    fields: dict[str, np.ndarray],
    radial_nodes: np.ndarray,
    query: np.ndarray,
    time_points: int,
) -> dict[str, np.ndarray]:
    harmonics = np.arange(1, fields["uz"].shape[1] + 1)
    time_cycle = np.arange(time_points, dtype=float) / time_points
    basis = np.exp(1j * 2.0 * np.pi * np.outer(harmonics, time_cycle))
    return {
        key: np.real(interpolate_columns(radial_nodes, values, query) @ basis)
        for key, values in fields.items()
    }


def hierarchy_waveforms(
    case: Any, hierarchy: HarmonicHierarchy, config: Step4Config
) -> dict[str, np.ndarray | float]:
    fluid = FluidProperties()
    endothelium = EndothelialControlVolume()
    eta = endothelium.thickness_m / case.radius_m
    near_wall_r = np.linspace(1.0 - eta, 1.0, config.quadrature_nodes)
    fields = {
        "uz0": hierarchy.uz0,
        "ut1": hierarchy.ut1,
        "uz2": hierarchy.uz2,
        "oz1": hierarchy.oz1,
        "ot0": hierarchy.ot0,
        "ot2": hierarchy.ot2,
    }
    harmonics = np.arange(1, hierarchy.uz0.shape[1] + 1)
    time_cycle = np.arange(config.time_points, dtype=float) / config.time_points
    basis = np.exp(1j * 2.0 * np.pi * np.outer(harmonics, time_cycle))
    real = {
        key: np.real(interpolate_columns(hierarchy.r, values, near_wall_r) @ basis)
        for key, values in fields.items()
    }
    lamb0 = -real["uz0"] * real["ot0"]
    lamb2 = (
        real["ut1"] * real["oz1"]
        - real["uz2"] * real["ot0"]
        - real["uz0"] * real["ot2"]
    )
    velocity_scale = (
        case.pressure_gradient_scale_pa_per_m
        * case.radius_m**2
        / fluid.dynamic_viscosity_pa_s
    )
    force_scale = endothelium.area_m2 * fluid.density_kg_m3 * velocity_scale**2
    force0 = force_scale * np.trapezoid(lamb0, near_wall_r, axis=0)
    force2 = force_scale * np.trapezoid(lamb2, near_wall_r, axis=0)
    return {
        "time_cycle": time_cycle,
        "near_wall_r_star": near_wall_r,
        "lamb0": lamb0,
        "lamb2": lamb2,
        "force0_n": force0,
        "force2_n": force2,
        "force_scale_n": float(force_scale),
    }


def direct_waveforms(
    case: Any,
    fields: dict[str, np.ndarray],
    hierarchy: HarmonicHierarchy,
    config: Step4Config,
) -> dict[str, np.ndarray]:
    fluid = FluidProperties()
    endothelium = EndothelialControlVolume()
    eta = endothelium.thickness_m / case.radius_m
    near_wall_r = np.linspace(1.0 - eta, 1.0, config.quadrature_nodes)
    real = real_fields(fields, hierarchy.r, near_wall_r, config.time_points)
    lamb = real["ut"] * real["oz"] - real["uz"] * real["ot"]
    velocity_scale = (
        case.pressure_gradient_scale_pa_per_m
        * case.radius_m**2
        / fluid.dynamic_viscosity_pa_s
    )
    force_scale = endothelium.area_m2 * fluid.density_kg_m3 * velocity_scale**2
    return {
        "signed_n": force_scale * np.trapezoid(lamb, near_wall_r, axis=0),
        "exposure_n": force_scale * np.trapezoid(np.abs(lamb), near_wall_r, axis=0),
        "lamb": lamb,
    }


def fit_log_slope(epsilon: Iterable[float], response: Iterable[float]) -> float:
    epsilon_array = np.asarray(tuple(epsilon), dtype=float)
    response_array = np.asarray(tuple(response), dtype=float)
    if np.any(response_array <= 0.0):
        raise RuntimeError("cannot fit order to nonpositive response")
    return float(np.polyfit(np.log(epsilon_array), np.log(response_array), 1)[0])


def contiguous_valid_max(rows: Any, mask: np.ndarray) -> float:
    valid_max = 0.0
    for epsilon, valid in zip(
        rows["epsilon"].to_numpy(), np.asarray(mask, dtype=bool), strict=True
    ):
        if not valid:
            break
        valid_max = float(epsilon)
    return valid_max
