"""Typed immutable model inputs and preserved v2 artery cases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SolverMode = Literal["verified", "reproduction"]
ForceMode = Literal["signed", "magnitude", "outward_only", "inward_only"]
StressMode = Literal["signed", "magnitude", "positive_only", "negative_only"]

BOLTZMANN_J_PER_K = 1.380649e-23
V2_EXPECTED_BLOB_SHA = "9d61c237cda75df338ce0383038f7765c886f503"

@dataclass(frozen=True)
class ArteryCase:
    artery_id: str
    name: str
    radius_m: float
    pressure_gradient_scale_pa_per_m: float
    harmonic_coefficients: tuple[float, ...]

    def validate(self) -> None:
        if self.radius_m <= 0:
            raise ValueError("radius_m must be positive")
        if self.pressure_gradient_scale_pa_per_m <= 0:
            raise ValueError("pressure gradient scale must be positive")
        if len(self.harmonic_coefficients) != 6:
            raise ValueError("v2 preservation requires exactly six harmonics")
        if not np.all(np.isfinite(self.harmonic_coefficients)):
            raise ValueError("harmonic coefficients must be finite")


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
            raise ValueError("density must be positive")
        if self.kinematic_viscosity_m2_s <= 0:
            raise ValueError("kinematic viscosity must be positive")
        if self.fundamental_frequency_hz <= 0:
            raise ValueError("fundamental frequency must be positive")


@dataclass(frozen=True)
class EndothelialControlVolume:
    area_m2: float = 100e-12
    volume_m3: float = 1e-15

    @property
    def thickness_m(self) -> float:
        return self.volume_m3 / self.area_m2

    def validate(self) -> None:
        if self.area_m2 <= 0 or self.volume_m3 <= 0:
            raise ValueError("endothelial area and volume must be positive")


@dataclass(frozen=True)
class HydrodynamicConfig:
    radial_order: int = 150
    time_points: int = 2048
    quadrature_nodes: int = 256
    beta: float = 0.1
    gamma: float = 0.1
    delta: float = 1.0
    mode: SolverMode = "verified"

    def validate(self) -> None:
        if self.radial_order < 30:
            raise ValueError("radial_order must be at least 30")
        if self.time_points < 64:
            raise ValueError("time_points must be at least 64")
        if self.quadrature_nodes < 8:
            raise ValueError("quadrature_nodes must be at least 8")
        if self.mode not in ("verified", "reproduction"):
            raise ValueError("unknown solver mode")
        if self.delta - ((self.beta + self.gamma) / 2.0) ** 2 <= 0:
            raise ValueError("anisotropic constitutive sample violates positive dissipation")


@dataclass(frozen=True)
class SensorConfig:
    basal_probability: float = 0.01
    relaxation_time_s: float = 0.1
    transition_fraction: float = 0.5
    temperature_k: float = 310.15

    def validate(self) -> None:
        if not 0.0 < self.basal_probability < 1.0:
            raise ValueError("basal_probability must lie in (0,1)")
        if self.relaxation_time_s <= 0:
            raise ValueError("relaxation_time_s must be positive")
        if not 0.0 <= self.transition_fraction <= 1.0:
            raise ValueError("transition_fraction must lie in [0,1]")
        if self.temperature_k <= 0:
            raise ValueError("temperature_k must be positive")


V2_ARTERY_CASES: tuple[ArteryCase, ...] = (
    ArteryCase("aortic_root", "Aortic Root", 0.015, 9000.0, (1.00, 0.82, 0.54, 0.33, 0.24, 0.17)),
    ArteryCase("thoracic_aorta", "Thoracic Aorta", 0.012, 7000.0, (1.00, 0.76, 0.45, 0.28, 0.20, 0.12)),
    ArteryCase("femoral", "Femoral", 0.004, 6000.0, (1.00, 0.58, 0.10, -0.17, 0.05, 0.04)),
    ArteryCase("carotid", "Carotid", 0.0035, 6500.0, (1.00, 0.63, 0.31, 0.15, 0.10, 0.06)),
    ArteryCase("iliac", "Iliac", 0.0045, 5500.0, (1.00, 0.51, 0.12, -0.11, 0.05, 0.03)),
    ArteryCase("brachial", "Brachial", 0.002, 4000.0, (1.00, 0.49, 0.16, -0.05, 0.02, 0.01)),
)
