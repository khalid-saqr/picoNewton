from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES

from .perturbation_core import (
    _EPS,
    Step4Config,
    derive_hierarchy,
    full_harmonic_fields,
    relative_l2,
    rms,
)
from .perturbation_observables import (
    contiguous_valid_max,
    direct_waveforms,
    fit_log_slope,
    hierarchy_waveforms,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_step3_artifacts(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    gate_path = root / "step3_gate.json"
    manifest_path = root / "step3_manifest.json"
    if not gate_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Step 4 requires Step 3 gate and manifest")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise RuntimeError("Step 4 requires a passing Step 3 gate")
    if manifest.get("status") != "complete" or manifest.get("allowed_next_step") != 4:
        raise RuntimeError("Step 3 manifest does not authorize Step 4")
    for name, record in manifest.get("files", {}).items():
        path = root / name
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise RuntimeError(f"Step 3 artifact failed checksum validation: {name}")
    return {"passed": True, "gate": gate, "manifest": manifest, "root": str(root)}


def run_perturbative_hierarchy(
    output_root: str | Path,
    step3_root: str | Path,
    config: Step4Config | None = None,
    *,
    require_step3: bool = True,
) -> dict[str, Any]:
    if config is None:
        config = Step4Config()
    config.validate()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if require_step3:
        step3 = _validate_step3_artifacts(step3_root)
    else:
        step3 = {"passed": False, "development_skip": True}

    sweep_rows: list[dict[str, Any]] = []
    slope_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    step3_continuity_rows: list[dict[str, Any]] = []
    step3_archive = None
    if require_step3 and config.profile == "publication":
        archive_path = Path(step3["root"]) / "six_artery_waveforms.npz"
        if not archive_path.is_file():
            raise RuntimeError("publication Step 4 requires the Step 3 waveform archive")
        step3_archive = np.load(archive_path)

    for case in V2_ARTERY_CASES:
        hierarchy = derive_hierarchy(case, config)
        perturbation = hierarchy_waveforms(case, hierarchy, config)
        force0 = np.asarray(perturbation["force0_n"])
        force2 = np.asarray(perturbation["force2_n"])

        arrays[f"{case.artery_id}__time_cycle"] = np.asarray(perturbation["time_cycle"])
        arrays[f"{case.artery_id}__force0_n"] = force0
        arrays[f"{case.artery_id}__force2_n"] = force2
        arrays[f"{case.artery_id}__uz0"] = hierarchy.uz0
        arrays[f"{case.artery_id}__ut1"] = hierarchy.ut1
        arrays[f"{case.artery_id}__uz2"] = hierarchy.uz2

        coefficient_rows.append(
            {
                "artery_id": case.artery_id,
                "artery_name": case.name,
                "hierarchy_max_normalized_residual": hierarchy.max_residual,
                "uz0_l2": float(np.linalg.norm(hierarchy.uz0)),
                "ut1_l2": float(np.linalg.norm(hierarchy.ut1)),
                "uz2_l2": float(np.linalg.norm(hierarchy.uz2)),
                "force2_rms_n_per_epsilon2": rms(force2),
                "force2_peak_abs_n_per_epsilon2": float(np.max(np.abs(force2))),
            }
        )

        artery_rows: list[dict[str, Any]] = []
        ut_norms: list[float] = []
        uz_correction_norms: list[float] = []
        force_excess_norms: list[float] = []
        slope_eps: list[float] = []
        direct_at_point_one: np.ndarray | None = None

        for epsilon in config.epsilon_values:
            full_fields, full_residual = full_harmonic_fields(case, config, epsilon)
            full_waveforms = direct_waveforms(case, full_fields, hierarchy, config)
            signed_full = np.asarray(full_waveforms["signed_n"])
            exposure_full = np.asarray(full_waveforms["exposure_n"])
            signed_excess = signed_full - force0
            if np.isclose(epsilon, 0.1):
                direct_at_point_one = signed_full.copy()
            signed_predicted = epsilon**2 * force2
            lamb_approx = np.asarray(perturbation["lamb0"]) + epsilon**2 * np.asarray(
                perturbation["lamb2"]
            )
            force_scale = float(perturbation["force_scale_n"])
            near_wall_r = np.asarray(perturbation["near_wall_r_star"])
            exposure0 = force_scale * np.trapezoid(
                np.abs(np.asarray(perturbation["lamb0"])), near_wall_r, axis=0
            )
            exposure_predicted = force_scale * np.trapezoid(
                np.abs(lamb_approx), near_wall_r, axis=0
            )
            exposure_excess = exposure_full - exposure0
            exposure_excess_predicted = exposure_predicted - exposure0

            ut_coefficient_error = relative_l2(full_fields["ut"] / epsilon, hierarchy.ut1)
            uz_coefficient_error = relative_l2(
                (full_fields["uz"] - hierarchy.uz0) / epsilon**2, hierarchy.uz2
            )
            signed_waveform_error = relative_l2(signed_excess, signed_predicted)
            signed_rms_error = abs(rms(signed_predicted) - rms(signed_excess)) / max(
                rms(signed_excess), _EPS
            )
            signed_peak_error = abs(
                np.max(np.abs(signed_predicted)) - np.max(np.abs(signed_excess))
            ) / max(np.max(np.abs(signed_excess)), _EPS)
            exposure_waveform_error = relative_l2(
                exposure_excess, exposure_excess_predicted
            )

            row = {
                "artery_id": case.artery_id,
                "artery_name": case.name,
                "epsilon": epsilon,
                "full_max_normalized_residual": full_residual,
                "ut_first_order_coefficient_relative_l2": ut_coefficient_error,
                "uz_second_order_coefficient_relative_l2": uz_coefficient_error,
                "signed_excess_waveform_relative_l2": signed_waveform_error,
                "signed_excess_rms_relative_error": signed_rms_error,
                "signed_excess_peak_relative_error": signed_peak_error,
                "exposure_excess_waveform_relative_l2": exposure_waveform_error,
                "signed_excess_rms_n": rms(signed_excess),
                "signed_excess_peak_abs_n": float(np.max(np.abs(signed_excess))),
                "predicted_signed_excess_rms_n": rms(signed_predicted),
                "predicted_signed_excess_peak_abs_n": float(
                    np.max(np.abs(signed_predicted))
                ),
                "scaled_signed_excess_rms_n_per_epsilon2": rms(signed_excess)
                / epsilon**2,
            }
            sweep_rows.append(row)
            artery_rows.append(row)
            arrays[f"{case.artery_id}__signed_excess_eps_{epsilon:.3f}_n"] = signed_excess
            arrays[f"{case.artery_id}__signed_predicted_eps_{epsilon:.3f}_n"] = signed_predicted

            if epsilon <= config.slope_epsilon_max:
                slope_eps.append(epsilon)
                ut_norms.append(float(np.linalg.norm(full_fields["ut"])))
                uz_correction_norms.append(
                    float(np.linalg.norm(full_fields["uz"] - hierarchy.uz0))
                )
                force_excess_norms.append(rms(signed_excess))

        if step3_archive is not None:
            isotropic_key = f"{case.artery_id}__signed_isotropic_n"
            anisotropic_key = f"{case.artery_id}__signed_anisotropic_n"
            if isotropic_key not in step3_archive or anisotropic_key not in step3_archive:
                raise RuntimeError(f"Step 3 archive is incomplete for {case.artery_id}")
            step3_isotropic = np.asarray(step3_archive[isotropic_key])
            step3_anisotropic = np.asarray(step3_archive[anisotropic_key])
            if step3_isotropic.shape != force0.shape or direct_at_point_one is None:
                raise RuntimeError("Step 3 and Step 4 publication resolutions do not match")
            step3_continuity_rows.append(
                {
                    "artery_id": case.artery_id,
                    "artery_name": case.name,
                    "isotropic_waveform_relative_l2": relative_l2(force0, step3_isotropic),
                    "epsilon_0p1_waveform_relative_l2": relative_l2(
                        direct_at_point_one, step3_anisotropic
                    ),
                }
            )

        slope_rows.append(
            {
                "artery_id": case.artery_id,
                "artery_name": case.name,
                "epsilon_max_used": max(slope_eps),
                "ut_order": fit_log_slope(slope_eps, ut_norms),
                "uz_correction_order": fit_log_slope(slope_eps, uz_correction_norms),
                "signed_force_excess_order": fit_log_slope(slope_eps, force_excess_norms),
            }
        )

        plus_fields, plus_residual = full_harmonic_fields(
            case, config, config.parity_epsilon
        )
        minus_fields, minus_residual = full_harmonic_fields(
            case, config, -config.parity_epsilon
        )
        plus_force = direct_waveforms(case, plus_fields, hierarchy, config)["signed_n"]
        minus_force = direct_waveforms(case, minus_fields, hierarchy, config)["signed_n"]
        parity_rows.append(
            {
                "artery_id": case.artery_id,
                "artery_name": case.name,
                "epsilon_abs": config.parity_epsilon,
                "uz_even_relative_l2": relative_l2(plus_fields["uz"], minus_fields["uz"]),
                "ut_odd_relative_l2": relative_l2(plus_fields["ut"], -minus_fields["ut"]),
                "signed_force_even_relative_l2": relative_l2(plus_force, minus_force),
                "max_normalized_residual": max(plus_residual, minus_residual),
            }
        )

        artery_table = pd.DataFrame(artery_rows).sort_values("epsilon")
        force_mask = (
            (artery_table["signed_excess_waveform_relative_l2"] <= config.relative_error_limit)
            & (artery_table["signed_excess_rms_relative_error"] <= config.relative_error_limit)
            & (artery_table["signed_excess_peak_relative_error"] <= config.relative_error_limit)
        ).to_numpy()
        ut_mask = (
            artery_table["ut_first_order_coefficient_relative_l2"]
            <= config.relative_error_limit
        ).to_numpy()
        uz_mask = (
            artery_table["uz_second_order_coefficient_relative_l2"]
            <= config.relative_error_limit
        ).to_numpy()
        validity_rows.append(
            {
                "artery_id": case.artery_id,
                "artery_name": case.name,
                "force_valid_epsilon_max_1pct": contiguous_valid_max(artery_table, force_mask),
                "ut_valid_epsilon_max_1pct": contiguous_valid_max(artery_table, ut_mask),
                "uz_valid_epsilon_max_1pct": contiguous_valid_max(artery_table, uz_mask),
                "minimum_required_epsilon": config.minimum_valid_epsilon,
            }
        )

    sweep = pd.DataFrame(sweep_rows)
    slopes = pd.DataFrame(slope_rows)
    parity = pd.DataFrame(parity_rows)
    validity = pd.DataFrame(validity_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    step3_continuity = pd.DataFrame(step3_continuity_rows)

    sweep.to_csv(output_root / "epsilon_sweep.csv", index=False)
    slopes.to_csv(output_root / "order_slopes.csv", index=False)
    parity.to_csv(output_root / "parity_checks.csv", index=False)
    validity.to_csv(output_root / "validity_domains.csv", index=False)
    coefficients.to_csv(output_root / "perturbation_coefficients.csv", index=False)
    if not step3_continuity.empty:
        step3_continuity.to_csv(output_root / "step3_waveform_continuity.csv", index=False)
    np.savez_compressed(output_root / "perturbation_waveforms.npz", **arrays)

    required_domain = validity[
        [
            "force_valid_epsilon_max_1pct",
            "ut_valid_epsilon_max_1pct",
            "uz_valid_epsilon_max_1pct",
        ]
    ].min(axis=1)
    gates: dict[str, Any] = {
        "step": 4,
        "profile": config.profile,
        "publication_profile": config.profile == "publication",
        "step3_gate_consumed": bool(step3.get("passed")),
        "six_arteries_complete": len(validity) == 6,
        "step3_waveform_continuity_passed": bool(
            not step3_continuity.empty
            and step3_continuity[
                [
                    "isotropic_waveform_relative_l2",
                    "epsilon_0p1_waveform_relative_l2",
                ]
            ].to_numpy().max()
            <= 1e-12
        ) if config.profile == "publication" else False,
        "hierarchy_residual_passed": bool(
            coefficients["hierarchy_max_normalized_residual"].max() <= 1e-10
        ),
        "full_model_residual_passed": bool(
            max(
                sweep["full_max_normalized_residual"].max(),
                parity["max_normalized_residual"].max(),
            )
            <= 1e-10
        ),
        "parity_passed": bool(
            parity[
                [
                    "uz_even_relative_l2",
                    "ut_odd_relative_l2",
                    "signed_force_even_relative_l2",
                ]
            ].to_numpy().max()
            <= 1e-10
        ),
        "ut_order_passed": bool(slopes["ut_order"].between(0.98, 1.02).all()),
        "uz_order_passed": bool(
            slopes["uz_correction_order"].between(1.95, 2.05).all()
        ),
        "force_order_passed": bool(
            slopes["signed_force_excess_order"].between(1.95, 2.05).all()
        ),
        "minimum_valid_domain_passed": bool(
            (required_domain >= config.minimum_valid_epsilon - 1e-15).all()
        ),
        "exposure_not_used_as_exact_kernel": True,
        "interaction_kernel_or_inversion_run": False,
    }
    required = [
        "publication_profile",
        "step3_gate_consumed",
        "six_arteries_complete",
        "step3_waveform_continuity_passed",
        "hierarchy_residual_passed",
        "full_model_residual_passed",
        "parity_passed",
        "ut_order_passed",
        "uz_order_passed",
        "force_order_passed",
        "minimum_valid_domain_passed",
        "exposure_not_used_as_exact_kernel",
    ]
    gates["passed"] = all(bool(gates[name]) for name in required)
    (output_root / "step4_gate.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True), encoding="utf-8"
    )

    output_names = [
        "epsilon_sweep.csv",
        "order_slopes.csv",
        "parity_checks.csv",
        "validity_domains.csv",
        "perturbation_coefficients.csv",
        "step3_waveform_continuity.csv",
        "perturbation_waveforms.npz",
        "step4_gate.json",
    ]
    manifest: dict[str, Any] = {
        "step": 4,
        "status": "complete" if gates["passed"] else "failed",
        "profile": config.profile,
        "step3_gate_consumed": bool(step3.get("passed")),
        "scientific_scope": "weak_anisotropy_perturbation_only",
        "allowed_next_step": 5 if gates["passed"] else None,
        "gates": gates,
        "files": {},
    }
    if config.profile != "publication":
        output_names.remove("step3_waveform_continuity.csv")
    for name in output_names:
        path = output_root / name
        manifest["files"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (output_root / "step4_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    return {
        "manifest": manifest,
        "sweep": sweep,
        "slopes": slopes,
        "parity": parity,
        "validity": validity,
        "coefficients": coefficients,
        "step3_continuity": step3_continuity,
    }
