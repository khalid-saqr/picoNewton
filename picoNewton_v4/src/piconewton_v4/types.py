"""Immutable inputs for the standalone v4 hydrodynamic layer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np


@dataclass(frozen=True)
class FluidProperties:
    density_kg_m3: float = 1060.0
    kinematic_viscosity_m2_s: float = 3.5e-6
    fundamental_frequency_hz: float = 1.2

    @property
    def dynamic_viscosity_pa_s(self) -> float:
        return self.density_kg_m3 * self.kinematic_viscosity_m2_s

    @property
    def angular_frequency_rad_s(self) -> float:
        return 2.0 * np.pi * self.fundamental_frequency_hz

    def validate(self) -> None:
        if self.density_kg_m3 <= 0:
            raise ValueError("density_kg_m3 must be positive")
        if self.kinematic_viscosity_m2_s <= 0:
            raise ValueError("kinematic_viscosity_m2_s must be positive")
        if self.fundamental_frequency_hz <= 0:
            raise ValueError("fundamental_frequency_hz must be positive")


@dataclass(frozen=True)
class EndothelialControlVolume:
    area_m2: float = 100e-12
    volume_m3: float = 1e-15

    @property
    def thickness_m(self) -> float:
        return self.volume_m3 / self.area_m2

    def validate(self) -> None:
        if self.area_m2 <= 0 or self.volume_m3 <= 0:
            raise ValueError("control-volume dimensions must be positive")


@dataclass(frozen=True)
class HydrodynamicConfig:
    radial_order: int = 150
    time_points: int = 2048
    near_wall_nodes: int = 256
    beta: float = 0.1
    gamma: float = 0.1
    delta: float = 1.0

    def validate(self) -> None:
        if self.radial_order < 30:
            raise ValueError("radial_order must be at least 30")
        if self.time_points < 64:
            raise ValueError("time_points must be at least 64")
        if self.near_wall_nodes < 8:
            raise ValueError("near_wall_nodes must be at least 8")
        if self.delta - ((self.beta + self.gamma) / 2.0) ** 2 <= 0:
            raise ValueError("constitutive parameters violate positive dissipation")


@dataclass(frozen=True)
class ArteryCase:
    artery_id: str
    name: str
    radius_m: float
    womersley_alpha_reference: float
    pressure_gradient_scale_pa_per_m: float
    harmonic_coefficients: tuple[float, ...]

    def validate(self) -> None:
        if not self.artery_id:
            raise ValueError("artery_id is required")
        if self.radius_m <= 0:
            raise ValueError("radius_m must be positive")
        if self.womersley_alpha_reference <= 0:
            raise ValueError("womersley_alpha_reference must be positive")
        if self.pressure_gradient_scale_pa_per_m <= 0:
            raise ValueError("pressure_gradient_scale_pa_per_m must be positive")
        if len(self.harmonic_coefficients) != 6:
            raise ValueError("exactly six harmonics are required")
        if not np.all(np.isfinite(self.harmonic_coefficients)):
            raise ValueError("harmonic coefficients must be finite")


def load_artery_cases(path: Path) -> tuple[ArteryCase, ...]:
    rows: list[ArteryCase] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case = ArteryCase(
                artery_id=row["artery_id"],
                name=row["artery"],
                radius_m=float(row["radius_m"]),
                womersley_alpha_reference=float(row["womersley_alpha"]),
                pressure_gradient_scale_pa_per_m=float(row["pressure_gradient_scale_pa_per_m"]),
                harmonic_coefficients=tuple(float(row[f"h{i}"]) for i in range(1, 7)),
            )
            case.validate()
            rows.append(case)
    if len(rows) != 6:
        raise ValueError(f"expected six artery cases, found {len(rows)}")
    if len({case.artery_id for case in rows}) != 6:
        raise ValueError("artery identifiers must be unique")
    return tuple(rows)
