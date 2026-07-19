"""Immutable inputs for the standalone hydrodynamic layer."""
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
        if self.density_kg_m3 <= 0 or self.kinematic_viscosity_m2_s <= 0 or self.fundamental_frequency_hz <= 0:
            raise ValueError("fluid properties must be positive")

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
    radial_order: int = 48
    time_points: int = 256
    near_wall_nodes: int = 48
    beta: float = 0.1
    gamma: float = 0.1
    delta: float = 1.0
    def validate(self) -> None:
        if self.radial_order < 30 or self.time_points < 64 or self.near_wall_nodes < 8:
            raise ValueError("insufficient hydrodynamic resolution")
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
        if not self.artery_id or self.radius_m <= 0 or self.womersley_alpha_reference <= 0 or self.pressure_gradient_scale_pa_per_m <= 0:
            raise ValueError("invalid artery case")
        if len(self.harmonic_coefficients) != 6 or not np.all(np.isfinite(self.harmonic_coefficients)):
            raise ValueError("exactly six finite harmonics are required")

def load_artery_cases(path: Path) -> tuple[ArteryCase, ...]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case = ArteryCase(
                artery_id=row["artery_id"], name=row["artery"], radius_m=float(row["radius_m"]),
                womersley_alpha_reference=float(row["womersley_alpha"]),
                pressure_gradient_scale_pa_per_m=float(row["pressure_gradient_scale_pa_per_m"]),
                harmonic_coefficients=tuple(float(row[f"h{i}"]) for i in range(1, 7)),
            )
            case.validate(); rows.append(case)
    if len(rows) != 6 or len({x.artery_id for x in rows}) != 6:
        raise ValueError("expected six unique artery cases")
    return tuple(rows)
