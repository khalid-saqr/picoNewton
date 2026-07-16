"""Shared helpers for workflow modules."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .model import (
    ArteryCase, EndothelialControlVolume, FluidProperties, HydrodynamicConfig,
    SensorConfig, compute_hydrodynamics, periodic_sensor_solution,
)

DEFAULT_SEED = 20260716
NUMERICAL_SENSOR_UNCERTAINTY = 1e-4

def run_hydrodynamic_cases(
    cases: Sequence[ArteryCase],
    config: HydrodynamicConfig,
    fluid: FluidProperties = FluidProperties(),
    endothelium: EndothelialControlVolume = EndothelialControlVolume(),
    *,
    include_fields: bool = False,
) -> dict[str, dict[str, Any]]:
    return {
        case.artery_id: compute_hydrodynamics(
            case,
            config,
            fluid,
            endothelium,
            include_near_wall_fields=include_fields,
        )
        for case in cases
    }


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def _phase_distance(a: float, b: float) -> float:
    raw = abs(a - b)
    return float(min(raw, 1.0 - raw))


def _sensor_from_work(
    work: np.ndarray,
    fluid: FluidProperties,
    sensor: SensorConfig,
) -> tuple[np.ndarray, float]:
    return periodic_sensor_solution(work, fluid.fundamental_frequency_hz, sensor)
