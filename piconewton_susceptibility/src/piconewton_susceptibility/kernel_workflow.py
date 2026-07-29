from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES

from .kernel_case import analyse_case
from .kernel_core import Step5Config
from .kernel_workflow_support import sha256, validate_step4_artifacts


def run_harmonic_kernel(
    output_root: str | Path,
    step4_root: str | Path,
    config: Step5Config | None = None,
    *,
    require_step4: bool = True,
) -> dict[str, Any]:
    if config is None:
        config = Step5Config()
    config.validate()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if require_step4:
        step4 = validate_step4_artifacts(step4_root)
    else:
        step4 = {"passed": False, "development_skip": True}

    step4_archive = None
    if require_step4 and config.profile == "publication":
        archive_path = Path(step4["root"]) / "perturbation_waveforms.npz"
        if not archive_path.is_file():
            raise RuntimeError("publication Step 5 requires the Step 4 waveform archive")
        step4_archive = np.load(archive_path)

    buckets: dict[str, list[dict[str, Any]]] = {
        "closure": [],
        "spectra": [],
        "kernels": [],
        "pairs": [],
        "dominant": [],
        "selection": [],
        "step4": [],
        "asymptotic": [],
    }
    arrays: dict[str, np.ndarray] = {}
    for case in V2_ARTERY_CASES:
        case_result = analyse_case(case, config, step4_archive)
        for name in buckets:
            buckets[name].extend(case_result[name])
        arrays.update(case_result["arrays"])

    closure = pd.DataFrame(buckets["closure"])
    spectra = pd.DataFrame(buckets["spectra"])
    kernels = pd.DataFrame(buckets["kernels"])
    pairs = pd.DataFrame(buckets["pairs"])
    dominant = pd.DataFrame(buckets["dominant"])
    selection = pd.DataFrame(buckets["selection"])
    step4_continuity = pd.DataFrame(buckets["step4"])
    asymptotic = pd.DataFrame(buckets["asymptotic"])

    closure.to_csv(output_root / "kernel_closure.csv", index=False)
    spectra.to_csv(output_root / "force_spectra.csv", index=False)
    kernels.to_csv(output_root / "kernel_entries.csv", index=False)
    pairs.to_csv(output_root / "pair_contributions.csv", index=False)
    dominant.to_csv(output_root / "dominant_pairs.csv", index=False)
    selection.to_csv(output_root / "selection_rule_controls.csv", index=False)
    asymptotic.to_csv(output_root / "kernel_asymptotic_closure.csv", index=False)
    if config.profile == "publication":
        step4_continuity.to_csv(output_root / "step4_kernel_continuity.csv", index=False)
    np.savez_compressed(output_root / "kernel_archive.npz", **arrays)

    outside = selection[selection["outside_allowed"]]
    allowed = selection[selection["allowed"]]
    allowed_group = allowed.groupby(["kernel_type", "control", "q"])["relative_to_max"].max()
    required_nonzero = bool((allowed_group > 1e-10).all())
    gates: dict[str, Any] = {
        "step": 5,
        "profile": config.profile,
        "publication_profile": config.profile == "publication",
        "step4_gate_consumed": bool(step4.get("passed")),
        "six_arteries_complete": closure["artery_id"].nunique() == 6,
        "exact_kernel_closure_passed": bool(
            closure[closure["kernel_type"].str.startswith("exact_excess")][
                ["waveform_relative_l2", "spectrum_relative_l2"]
            ].to_numpy().max()
            <= config.closure_tolerance
        ),
        "second_order_kernel_closure_passed": bool(
            closure[closure["kernel_type"].str.startswith("second_order")][
                ["waveform_relative_l2", "spectrum_relative_l2"]
            ].to_numpy().max()
            <= config.closure_tolerance
        ),
        "hermitian_and_real_reconstruction_passed": bool(
            closure[
                ["reconstruction_imaginary_relative_max", "hermitian_relative_max"]
            ].to_numpy().max()
            <= config.closure_tolerance
        ),
        "response_residual_passed": bool(
            closure["max_normalized_response_residual"].max() <= 1e-10
        ),
        "selection_rule_support_passed": bool(
            (outside["relative_to_max"].max() if len(outside) else 0.0)
            <= config.selection_tolerance
            and required_nonzero
        ),
        "kernel_asymptotic_2pct_passed": bool(
            asymptotic["scaled_exact_vs_second_order_kernel_relative_l2"].max()
            <= 0.02
        ),
        "step4_force2_continuity_passed": bool(
            step4_continuity["force2_waveform_relative_l2"].max() <= 1e-12
        )
        if config.profile == "publication"
        else False,
        "dc_sum_difference_and_doubling_present": required_nonzero,
        "exposure_used_in_kernel": False,
        "susceptibility_or_threshold_inversion_run": False,
    }
    required = [
        "publication_profile",
        "step4_gate_consumed",
        "six_arteries_complete",
        "exact_kernel_closure_passed",
        "second_order_kernel_closure_passed",
        "hermitian_and_real_reconstruction_passed",
        "response_residual_passed",
        "selection_rule_support_passed",
        "kernel_asymptotic_2pct_passed",
        "step4_force2_continuity_passed",
        "dc_sum_difference_and_doubling_present",
    ]
    gates["passed"] = all(bool(gates[name]) for name in required)
    (output_root / "step5_gate.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True), encoding="utf-8"
    )

    output_names = [
        "kernel_closure.csv",
        "force_spectra.csv",
        "kernel_entries.csv",
        "pair_contributions.csv",
        "dominant_pairs.csv",
        "selection_rule_controls.csv",
        "kernel_asymptotic_closure.csv",
        "kernel_archive.npz",
        "step5_gate.json",
    ]
    if config.profile == "publication":
        output_names.append("step4_kernel_continuity.csv")
    manifest: dict[str, Any] = {
        "step": 5,
        "status": "complete" if gates["passed"] else "failed",
        "profile": config.profile,
        "step4_gate_consumed": bool(step4.get("passed")),
        "scientific_scope": "exact_signed_harmonic_interaction_kernel_only",
        "exact_epsilon": config.exact_epsilon,
        "allowed_next_step": 6 if gates["passed"] else None,
        "gates": gates,
        "files": {},
    }
    for name in output_names:
        path = output_root / name
        manifest["files"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (output_root / "step5_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "closure": closure,
        "spectra": spectra,
        "kernels": kernels,
        "pairs": pairs,
        "dominant": dominant,
        "selection": selection,
        "asymptotic": asymptotic,
        "step4_continuity": step4_continuity,
    }
