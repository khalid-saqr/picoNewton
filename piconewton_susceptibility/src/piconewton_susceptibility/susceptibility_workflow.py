from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES

from .susceptibility_case import analyse_case
from .susceptibility_core import Step6Config
from .susceptibility_support import sha256, validate_step4_artifacts, validate_step5_artifacts


def run_susceptibility_inversion(
    output_root: str | Path,
    step5_root: str | Path,
    step4_root: str | Path,
    config: Step6Config | None = None,
    *,
    require_prior_steps: bool = True,
) -> dict[str, Any]:
    if config is None:
        config = Step6Config()
    config.validate()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if require_prior_steps:
        step5 = validate_step5_artifacts(step5_root)
        step4 = validate_step4_artifacts(step4_root)
    else:
        step5 = {"passed": False, "development_skip": True}
        step4 = {"passed": False, "development_skip": True}

    step5_archive = None
    validity_by_artery: dict[str, float] = {}
    if require_prior_steps and config.profile == "publication":
        archive_path = Path(step5["root"]) / "kernel_archive.npz"
        validity_path = Path(step4["root"]) / "validity_domains.csv"
        if not archive_path.is_file() or not validity_path.is_file():
            raise RuntimeError("publication Step 6 requires Step 4 and Step 5 scientific archives")
        step5_archive = np.load(archive_path)
        validity = pd.read_csv(validity_path)
        validity_by_artery = {
            str(row.artery_id): float(row.force_valid_epsilon_max_1pct)
            for row in validity.itertuples(index=False)
        }

    buckets: dict[str, list[dict[str, Any]]] = {
        "native": [],
        "harmonics": [],
        "exact": [],
        "scale": [],
        "inverse": [],
        "critical": [],
        "continuity": [],
    }
    arrays: dict[str, np.ndarray] = {}
    for case in V2_ARTERY_CASES:
        force_valid_max = validity_by_artery.get(case.artery_id, 0.0)
        if config.profile == "publication" and force_valid_max <= 0.0:
            raise RuntimeError(f"missing Step 4 validity domain for {case.artery_id}")
        if config.profile != "publication":
            force_valid_max = min(max(config.validation_epsilons), 0.08)
        result = analyse_case(case, config, step5_archive, force_valid_max)
        for name in buckets:
            buckets[name].extend(result[name])
        arrays.update(result["arrays"])

    native = pd.DataFrame(buckets["native"])
    harmonics = pd.DataFrame(buckets["harmonics"])
    exact = pd.DataFrame(buckets["exact"])
    scale_table = pd.DataFrame(buckets["scale"])
    inverse = pd.DataFrame(buckets["inverse"])
    critical = pd.DataFrame(buckets["critical"])
    continuity = pd.DataFrame(buckets["continuity"])

    native.to_csv(output_root / "native_susceptibility.csv", index=False)
    harmonics.to_csv(output_root / "harmonic_susceptibility.csv", index=False)
    exact.to_csv(output_root / "exact_susceptibility_validation.csv", index=False)
    scale_table.to_csv(output_root / "pressure_scale_invariance.csv", index=False)
    inverse.to_csv(output_root / "inverse_verification.csv", index=False)
    critical.to_csv(output_root / "critical_anisotropy.csv", index=False)
    if config.profile == "publication":
        continuity.to_csv(output_root / "step5_susceptibility_continuity.csv", index=False)
    np.savez_compressed(output_root / "susceptibility_archive.npz", **arrays)

    statuses = {
        "unreachable_within_validated_domain",
        "unreachable_and_perturbative_estimate_out_of_domain",
        "full_model_crossing_found",
        "full_crossing_found_but_perturbative_estimate_out_of_domain",
        "nonmonotonic_within_validated_domain",
        "unreachable_and_formal_estimate_constitutively_inadmissible",
        "full_crossing_found_but_formal_estimate_inadmissible",
    }
    exact_gate = exact[exact["within_step4_valid_domain"]]
    gates: dict[str, Any] = {
        "step": 6,
        "profile": config.profile,
        "publication_profile": config.profile == "publication",
        "step5_gate_consumed": bool(step5.get("passed")),
        "step4_gate_consumed": bool(step4.get("passed")),
        "six_arteries_complete": native["artery_id"].nunique() == 6,
        "susceptibility_kernel_continuity_passed": bool(
            continuity[
                [
                    "force2_waveform_relative_l2",
                    "force2_spectrum_relative_l2",
                    "dimensional_reconstruction_relative_l2",
                ]
            ].to_numpy().max()
            <= config.closure_tolerance
        )
        if config.profile == "publication"
        else False,
        "exact_archive_portability_passed": bool(
            continuity[
                [
                    "exact_epsilon_0p1_waveform_relative_l2",
                    "exact_epsilon_0p1_spectrum_relative_l2",
                ]
            ].to_numpy().max()
            <= config.cross_environment_exact_tolerance
        )
        if config.profile == "publication"
        else False,
        "parseval_closure_passed": bool(
            native["phi_parseval_relative_error"].max() <= config.closure_tolerance
        ),
        "pressure_scale_invariance_passed": bool(
            scale_table[
                ["waveform_relative_l2", "spectrum_relative_l2", "force_scale_ratio_error"]
            ].to_numpy().max()
            <= config.closure_tolerance
        ),
        "exact_validation_within_step4_domains": bool(
            exact_gate[
                ["waveform_relative_l2", "rms_relative_error", "peak_relative_error"]
            ].to_numpy().max()
            <= config.exact_validation_relative_limit
        ),
        "inverse_formula_verification_passed": bool(
            inverse["perturbative_relative_error"].max()
            <= config.inverse_estimate_relative_limit
            and inverse["full_model_absolute_error"].max()
            <= config.inverse_root_absolute_tolerance * 2.0
            and (inverse["full_model_status"] == "full_model_crossing_found").all()
        ),
        "published_benchmarks_frozen": tuple(sorted(critical["benchmark_pn"].unique()))
        == (1.0, 10.0),
        "benchmark_states_complete": bool(
            len(critical) == 24 and set(critical["status"]).issubset(statuses)
        ),
        "constitutive_admissibility_flag_complete": bool(
            critical["formal_estimate_constitutively_admissible"].notna().all()
            and critical.loc[
                ~critical["formal_estimate_constitutively_admissible"], "status"
            ].str.contains("inadmissible").all()
        ),
        "no_silent_extrapolation": bool(
            critical.loc[
                ~critical["perturbative_estimate_in_domain"], "full_model_crossing"
            ].isna().all()
        ),
        "crossed_waveforms_or_reduction_run": False,
        "exposure_kernel_or_biological_threshold_used": False,
    }
    required = [
        "publication_profile",
        "step5_gate_consumed",
        "step4_gate_consumed",
        "six_arteries_complete",
        "susceptibility_kernel_continuity_passed",
        "exact_archive_portability_passed",
        "parseval_closure_passed",
        "pressure_scale_invariance_passed",
        "exact_validation_within_step4_domains",
        "inverse_formula_verification_passed",
        "published_benchmarks_frozen",
        "benchmark_states_complete",
        "constitutive_admissibility_flag_complete",
        "no_silent_extrapolation",
    ]
    gates["passed"] = all(bool(gates[name]) for name in required)
    (output_root / "step6_gate.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True), encoding="utf-8"
    )

    output_names = [
        "native_susceptibility.csv",
        "harmonic_susceptibility.csv",
        "exact_susceptibility_validation.csv",
        "pressure_scale_invariance.csv",
        "inverse_verification.csv",
        "critical_anisotropy.csv",
        "susceptibility_archive.npz",
        "step6_gate.json",
    ]
    if config.profile == "publication":
        output_names.append("step5_susceptibility_continuity.csv")
    manifest: dict[str, Any] = {
        "step": 6,
        "status": "complete" if gates["passed"] else "failed",
        "profile": config.profile,
        "step5_gate_consumed": bool(step5.get("passed")),
        "step4_gate_consumed": bool(step4.get("passed")),
        "scientific_scope": "native_waveform_susceptibility_and_critical_anisotropy",
        "force_benchmarks_pn": list(config.force_benchmarks_pn),
        "allowed_next_step": 7 if gates["passed"] else None,
        "gates": gates,
        "files": {},
    }
    for name in output_names:
        path = output_root / name
        manifest["files"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (output_root / "step6_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "native": native,
        "harmonics": harmonics,
        "exact": exact,
        "scale": scale_table,
        "inverse_verification": inverse,
        "critical": critical,
        "continuity": continuity,
    }
