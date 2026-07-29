from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from piconewton_v3 import (
    EndothelialControlVolume,
    FluidProperties,
    HydrodynamicConfig,
    compute_hydrodynamics,
)
from piconewton_v3.hydrodynamics import WomersleySolver

from .susceptibility_core import Step6Config, alpha_for_case, rms


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_artifacts(
    root: str | Path,
    *,
    step: int,
    expected_next: int,
    gate_name: str,
    manifest_name: str,
) -> dict[str, Any]:
    root = Path(root).resolve()
    gate_path = root / gate_name
    manifest_path = root / manifest_name
    if not gate_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Step 6 requires Step {step} gate and manifest")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise RuntimeError(f"Step 6 requires a passing Step {step} gate")
    if manifest.get("status") != "complete" or manifest.get("allowed_next_step") != expected_next:
        raise RuntimeError(f"Step {step} manifest does not authorize Step {expected_next}")
    for name, record in manifest.get("files", {}).items():
        path = root / name
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise RuntimeError(f"Step {step} artifact failed checksum validation: {name}")
    return {"passed": True, "gate": gate, "manifest": manifest, "root": str(root)}


def validate_step5_artifacts(root: str | Path) -> dict[str, Any]:
    return validate_artifacts(
        root,
        step=5,
        expected_next=6,
        gate_name="step5_gate.json",
        manifest_name="step5_manifest.json",
    )


def validate_step4_artifacts(root: str | Path) -> dict[str, Any]:
    return validate_artifacts(
        root,
        step=4,
        expected_next=5,
        gate_name="step4_gate.json",
        manifest_name="step4_manifest.json",
    )


def isotropic_normalizers(case: Any, config: Step6Config) -> dict[str, float]:
    result = compute_hydrodynamics(
        case,
        HydrodynamicConfig(
            radial_order=config.radial_order,
            time_points=config.time_points,
            quadrature_nodes=config.quadrature_nodes,
            beta=0.0,
            gamma=0.0,
            delta=1.0,
            mode="verified",
        ),
        FluidProperties(),
        EndothelialControlVolume(),
    )
    area = EndothelialControlVolume().area_m2
    fluid = FluidProperties()
    solver = WomersleySolver(config.radial_order, "verified")
    alpha = alpha_for_case(case)
    wall_shear_h = []
    for harmonic, coefficient in enumerate(case.harmonic_coefficients, start=1):
        uz, _ut, _residual = solver.solve_harmonic(
            alpha, harmonic, coefficient, 0.0, 0.0, 1.0
        )
        d_uz_wall = (solver.D @ uz)[-1]
        velocity_scale = (
            case.pressure_gradient_scale_pa_per_m
            * case.radius_m**2
            / fluid.dynamic_viscosity_pa_s
        )
        wall_shear_h.append(
            d_uz_wall * fluid.dynamic_viscosity_pa_s * velocity_scale / case.radius_m
        )
    harmonics = np.arange(1, len(wall_shear_h) + 1)
    time_cycle = np.arange(config.time_points, dtype=float) / config.time_points
    temporal = np.exp(1j * 2.0 * np.pi * np.outer(harmonics, time_cycle))
    wall_shear_pa = np.real(np.asarray(wall_shear_h) @ temporal)
    return {
        "isotropic_signed_rms_n": rms(np.asarray(result["force_signed_n"])),
        "isotropic_signed_peak_abs_n": float(np.max(np.abs(result["force_signed_n"]))),
        "isotropic_wss_force_rms_n": area * rms(wall_shear_pa),
        "isotropic_wss_force_peak_abs_n": area * float(np.max(np.abs(wall_shear_pa))),
    }
