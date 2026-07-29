from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from piconewton_v3 import (
    V2_ARTERY_CASES,
    EndothelialControlVolume,
    FluidProperties,
    HydrodynamicConfig,
    compute_hydrodynamics,
    isotropic_validation,
)

from .validation import validate_bootstrap_artifacts

EXPECTED_ALPHA = {
    "aortic_root": 22.03,
    "thoracic_aorta": 17.62,
    "femoral": 5.87,
    "carotid": 5.14,
    "iliac": 6.61,
    "brachial": 2.94,
}


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def _relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-30))


def _high_harmonic_fraction(values: np.ndarray, input_harmonics: int = 6) -> float:
    spectrum = np.fft.rfft(np.asarray(values, dtype=float) - np.mean(values))
    power = np.abs(spectrum) ** 2
    return float(np.sum(power[input_harmonics + 1 :]) / max(np.sum(power[1:]), 1e-30))


def _metrics(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_rms_n": _rms(values),
        f"{prefix}_peak_abs_n": float(np.max(np.abs(values))),
        f"{prefix}_mean_n": float(np.mean(values)),
        f"{prefix}_outward_duty": float(np.mean(values > 0.0)),
        f"{prefix}_inward_duty": float(np.mean(values < 0.0)),
        f"{prefix}_high_harmonic_fraction": _high_harmonic_fraction(values),
    }


def _mechanics_closure(result: dict[str, Any], radius_m: float) -> float:
    uz = np.asarray(result["u_z_m_s"])
    ut = np.asarray(result["u_theta_m_s"])
    omega_z = np.asarray(result["omega_z_s_inv"])
    omega_theta = np.asarray(result["omega_theta_s_inv"])
    radius = np.asarray(result["near_wall_r_star"])[:, None] * radius_m
    lamb = np.asarray(result["lamb_r_m_s2"])
    kinetic_gradient_plus_centrifugal = (
        uz * (-omega_theta)
        + ut * (omega_z - ut / radius)
        + ut**2 / radius
    )
    return _relative_l2(lamb, kinetic_gradient_plus_centrifugal)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_historical_baseline() -> dict[str, Any]:
    resource = files("piconewton_susceptibility").joinpath(
        "data/historical_mode_baseline.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Step3Config:
    profile: str = "publication"
    radial_order: int = 150
    time_points: int = 2048
    quadrature_nodes: int = 256
    beta: float = 0.1
    gamma: float = 0.1
    delta: float = 1.0
    radial_checks: tuple[int, ...] = (120, 180)
    time_checks: tuple[int, ...] = (1024, 4096)
    quadrature_checks: tuple[int, ...] = (128, 512)

    def validate(self) -> None:
        if self.profile not in {"quick", "publication"}:
            raise ValueError("profile must be quick or publication")
        if self.radial_order < 30 or self.time_points < 64 or self.quadrature_nodes < 8:
            raise ValueError("invalid numerical resolution")
        if self.delta - ((self.beta + self.gamma) / 2.0) ** 2 <= 0.0:
            raise ValueError("constitutive sample violates positive dissipation")


def _run_case(
    case: Any,
    config: Step3Config,
    *,
    mode: str = "verified",
    isotropic: bool = False,
    radial_order: int | None = None,
    time_points: int | None = None,
    quadrature_nodes: int | None = None,
    fields: bool = False,
) -> dict[str, Any]:
    hydro = HydrodynamicConfig(
        radial_order=radial_order or config.radial_order,
        time_points=time_points or config.time_points,
        quadrature_nodes=quadrature_nodes or config.quadrature_nodes,
        beta=0.0 if isotropic else config.beta,
        gamma=0.0 if isotropic else config.gamma,
        delta=1.0 if isotropic else config.delta,
        mode=mode,
    )
    return compute_hydrodynamics(
        case,
        hydro,
        FluidProperties(),
        EndothelialControlVolume(),
        include_near_wall_fields=fields,
    )


def _primary_vector(full: dict[str, Any], isotropic: dict[str, Any]) -> np.ndarray:
    signed_excess = np.asarray(full["force_signed_n"]) - np.asarray(isotropic["force_signed_n"])
    exposure_excess = np.asarray(full["force_exposure_n"]) - np.asarray(
        isotropic["force_exposure_n"]
    )
    return np.array(
        [
            _rms(full["force_signed_n"]),
            float(np.max(np.abs(full["force_signed_n"]))),
            _rms(full["force_exposure_n"]),
            float(np.max(full["force_exposure_n"])),
            _rms(signed_excess),
            float(np.max(np.abs(signed_excess))),
            _rms(exposure_excess),
            float(np.max(np.abs(exposure_excess))),
        ]
    )


def _historical_metrics(result: dict[str, Any]) -> dict[str, float]:
    return {
        "signed_rms_n": _rms(result["force_signed_n"]),
        "signed_peak_abs_n": float(np.max(np.abs(result["force_signed_n"]))),
        "exposure_rms_n": _rms(result["force_exposure_n"]),
        "exposure_peak_n": float(np.max(result["force_exposure_n"])),
    }


def run_parent_continuity(
    output_root: str | Path,
    step2_root: str | Path,
    config: Step3Config | None = None,
    *,
    require_step2: bool = True,
) -> dict[str, Any]:
    if config is None:
        config = Step3Config()
    config.validate()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if require_step2:
        step2 = validate_bootstrap_artifacts(step2_root, require_claim_bearing=True)
        if not step2.get("passed"):
            raise RuntimeError("Step 3 requires a passing Step 2 completion gate")
    else:
        step2 = {"passed": False, "development_skip": True}

    isotropic_rows = isotropic_validation(radial_order=config.radial_order)
    if not all(row["passed"] for row in isotropic_rows):
        raise RuntimeError("isotropic analytical verification failed")

    baseline = _load_historical_baseline()
    baseline_by_artery = {record["artery_id"]: record for record in baseline["records"]}
    baseline_tolerance = float(baseline["relative_tolerance"])

    summaries: list[dict[str, Any]] = []
    historical_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}

    for case in V2_ARTERY_CASES:
        full = _run_case(case, config, mode="verified", fields=True)
        isotropic = _run_case(case, config, mode="verified", isotropic=True, fields=True)
        historical = _run_case(case, config, mode="reproduction")

        signed_excess = np.asarray(full["force_signed_n"]) - np.asarray(
            isotropic["force_signed_n"]
        )
        exposure_excess = np.asarray(full["force_exposure_n"]) - np.asarray(
            isotropic["force_exposure_n"]
        )
        alpha_error = abs(full["alpha"] - EXPECTED_ALPHA[case.artery_id]) / EXPECTED_ALPHA[
            case.artery_id
        ]
        mechanics_error = max(
            _mechanics_closure(full, case.radius_m),
            _mechanics_closure(isotropic, case.radius_m),
        )

        row: dict[str, Any] = {
            "artery_id": case.artery_id,
            "artery_name": case.name,
            "radius_m": case.radius_m,
            "alpha_computed": full["alpha"],
            "alpha_published": EXPECTED_ALPHA[case.artery_id],
            "alpha_relative_error": alpha_error,
            "pressure_gradient_scale_pa_per_m": case.pressure_gradient_scale_pa_per_m,
            "max_backward_residual": max(
                full["max_normalized_backward_residual"],
                isotropic["max_normalized_backward_residual"],
            ),
            "mechanics_closure_relative_l2": mechanics_error,
            "anisotropic_exposure_nonnegative": bool(
                np.min(full["force_exposure_n"]) >= -1e-30
            ),
            "isotropic_exposure_nonnegative": bool(
                np.min(isotropic["force_exposure_n"]) >= -1e-30
            ),
        }
        row.update(_metrics(full["force_signed_n"], "anisotropic_signed"))
        row.update(_metrics(full["force_exposure_n"], "anisotropic_exposure"))
        row.update(_metrics(isotropic["force_signed_n"], "isotropic_signed"))
        row.update(_metrics(isotropic["force_exposure_n"], "isotropic_exposure"))
        row.update(_metrics(signed_excess, "anisotropic_signed_excess"))
        row.update(_metrics(exposure_excess, "anisotropic_exposure_excess"))
        row["signed_excess_fraction_of_isotropic_rms"] = row[
            "anisotropic_signed_excess_rms_n"
        ] / max(row["isotropic_signed_rms_n"], 1e-30)
        summaries.append(row)

        observed_historical = _historical_metrics(historical)
        reference_historical = baseline_by_artery[case.artery_id]
        baseline_errors = {
            key: abs(observed_historical[key] - reference_historical[key])
            / max(abs(reference_historical[key]), 1e-30)
            for key in observed_historical
        }
        historical_rows.append(
            {
                "artery_id": case.artery_id,
                "verified_vs_historical_signed_relative_l2": _relative_l2(
                    full["force_signed_n"], historical["force_signed_n"]
                ),
                "verified_vs_historical_exposure_relative_l2": _relative_l2(
                    full["force_exposure_n"], historical["force_exposure_n"]
                ),
                "historical_max_backward_residual": historical[
                    "max_normalized_backward_residual"
                ],
                "historical_baseline_max_relative_error": max(baseline_errors.values()),
                "historical_baseline_passed": max(baseline_errors.values())
                <= baseline_tolerance,
                "historical_role": "lineage_only",
            }
        )

        arrays[f"{case.artery_id}__time_cycle"] = np.asarray(full["time_cycle"])
        arrays[f"{case.artery_id}__signed_anisotropic_n"] = np.asarray(
            full["force_signed_n"]
        )
        arrays[f"{case.artery_id}__exposure_anisotropic_n"] = np.asarray(
            full["force_exposure_n"]
        )
        arrays[f"{case.artery_id}__signed_isotropic_n"] = np.asarray(
            isotropic["force_signed_n"]
        )
        arrays[f"{case.artery_id}__exposure_isotropic_n"] = np.asarray(
            isotropic["force_exposure_n"]
        )
        arrays[f"{case.artery_id}__signed_excess_n"] = signed_excess
        arrays[f"{case.artery_id}__exposure_excess_n"] = exposure_excess
        arrays[f"{case.artery_id}__signed_historical_n"] = np.asarray(
            historical["force_signed_n"]
        )

        base_vector = _primary_vector(full, isotropic)
        for dimension, values in (
            ("radial_order", config.radial_checks),
            ("time_points", config.time_checks),
            ("quadrature_nodes", config.quadrature_checks),
        ):
            for value in values:
                kwargs = {dimension: value}
                alternative_full = _run_case(case, config, mode="verified", **kwargs)
                alternative_isotropic = _run_case(
                    case, config, mode="verified", isotropic=True, **kwargs
                )
                relative_changes = np.abs(
                    _primary_vector(alternative_full, alternative_isotropic) - base_vector
                ) / np.maximum(np.abs(base_vector), 1e-30)
                convergence_rows.append(
                    {
                        "artery_id": case.artery_id,
                        "dimension": dimension,
                        "value": value,
                        "total_signed_rms_relative_change": relative_changes[0],
                        "total_signed_peak_relative_change": relative_changes[1],
                        "total_exposure_rms_relative_change": relative_changes[2],
                        "total_exposure_peak_relative_change": relative_changes[3],
                        "signed_excess_rms_relative_change": relative_changes[4],
                        "signed_excess_peak_relative_change": relative_changes[5],
                        "exposure_excess_rms_relative_change": relative_changes[6],
                        "exposure_excess_peak_relative_change": relative_changes[7],
                        "max_total_relative_change": float(np.max(relative_changes[:4])),
                        "max_excess_relative_change": float(np.max(relative_changes[4:])),
                    }
                )

    summary = pd.DataFrame(summaries)
    historical_table = pd.DataFrame(historical_rows)
    convergence = pd.DataFrame(convergence_rows)
    isotropic_table = pd.DataFrame(isotropic_rows)

    summary.to_csv(output_root / "six_artery_continuity.csv", index=False)
    historical_table.to_csv(output_root / "historical_mode_discrepancy.csv", index=False)
    convergence.to_csv(output_root / "convergence.csv", index=False)
    isotropic_table.to_csv(output_root / "isotropic_validation.csv", index=False)
    np.savez_compressed(output_root / "six_artery_waveforms.npz", **arrays)

    gates: dict[str, Any] = {
        "step": 3,
        "profile": config.profile,
        "publication_profile": config.profile == "publication",
        "step2_gate_consumed": bool(step2.get("passed")),
        "six_arteries_complete": len(summary) == 6,
        "isotropic_validation_passed": bool(isotropic_table["passed"].all()),
        "residual_gate_passed": bool(summary["max_backward_residual"].max() <= 1e-10),
        "alpha_inventory_passed": bool(summary["alpha_relative_error"].max() <= 5e-3),
        "mechanics_closure_passed": bool(
            summary["mechanics_closure_relative_l2"].max() <= 1e-12
        ),
        "exposure_nonnegative_passed": bool(
            summary["anisotropic_exposure_nonnegative"].all()
            and summary["isotropic_exposure_nonnegative"].all()
        ),
        "total_convergence_1pct_passed": bool(
            convergence["max_total_relative_change"].max() <= 0.01
        ),
        "excess_convergence_1pct_passed": bool(
            convergence["max_excess_relative_change"].max() <= 0.01
        ),
        "historical_baseline_passed": bool(
            historical_table["historical_baseline_passed"].all()
        ),
        "historical_mode_separated": bool(
            (historical_table["historical_role"] == "lineage_only").all()
        ),
        "perturbation_kernel_or_inversion_run": False,
    }
    required_gate_names = [
        "publication_profile",
        "step2_gate_consumed",
        "six_arteries_complete",
        "isotropic_validation_passed",
        "residual_gate_passed",
        "alpha_inventory_passed",
        "mechanics_closure_passed",
        "exposure_nonnegative_passed",
        "total_convergence_1pct_passed",
        "excess_convergence_1pct_passed",
        "historical_baseline_passed",
        "historical_mode_separated",
    ]
    gates["passed"] = all(bool(gates[name]) for name in required_gate_names)

    gate_path = output_root / "step3_gate.json"
    gate_path.write_text(json.dumps(gates, indent=2, sort_keys=True), encoding="utf-8")

    output_names = [
        "six_artery_continuity.csv",
        "historical_mode_discrepancy.csv",
        "convergence.csv",
        "isotropic_validation.csv",
        "six_artery_waveforms.npz",
        "step3_gate.json",
    ]
    manifest: dict[str, Any] = {
        "step": 3,
        "status": "complete" if gates["passed"] else "failed",
        "profile": config.profile,
        "step2_gate_consumed": bool(step2.get("passed")),
        "scientific_scope": "parent_continuity_only",
        "allowed_next_step": 4 if gates["passed"] else None,
        "gates": gates,
        "files": {},
    }
    for name in output_names:
        path = output_root / name
        manifest["files"][name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    manifest_path = output_root / "step3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "manifest": manifest,
        "summary": summary,
        "historical": historical_table,
        "convergence": convergence,
        "isotropic": isotropic_table,
    }
