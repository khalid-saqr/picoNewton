from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES

from .experiments_case import causal_family_rows, crossed_rows, native_control_rows
from .experiments_core import (
    Step7Config,
    additive_decomposition,
    causal_waveform_families,
    dimensionless_kernels,
    force_scale,
    native_eta,
    response_set,
)
from .experiments_support import matrix_effect_rows, sha256, validate_step6


def run_waveform_experiments(
    output_root: str | Path,
    step6_root: str | Path,
    config: Step7Config | None = None,
    *,
    require_step6: bool = True,
) -> dict[str, Any]:
    if config is None:
        config = Step7Config()
    config.validate()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    step6 = validate_step6(step6_root) if require_step6 else {"passed": False}

    native_reference = None
    if require_step6 and config.profile == "publication":
        native_path = Path(step6["root"]) / "native_susceptibility.csv"
        if not native_path.is_file():
            raise RuntimeError("publication Step 7 requires native_susceptibility.csv")
        native_reference = pd.read_csv(native_path).set_index("artery_id")

    responses = {case.artery_id: response_set(case, config) for case in V2_ARTERY_CASES}
    residual_rows = [
        {
            "vessel_id": case.artery_id,
            "max_normalized_response_residual": responses[case.artery_id].max_residual,
        }
        for case in V2_ARTERY_CASES
    ]
    matrix_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    continuity_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    matrix_values = {
        "hydrodynamic": np.zeros((6, 6), dtype=float),
        "physiological": np.zeros((6, 6), dtype=float),
    }
    kernel_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for matrix_type in ("hydrodynamic", "physiological"):
        for vessel_index, vessel in enumerate(V2_ARTERY_CASES):
            eta = config.eta_reference if matrix_type == "hydrodynamic" else native_eta(vessel)
            frequencies, kernel2, kernel_exact = dimensionless_kernels(
                vessel, responses[vessel.artery_id], eta, config
            )
            kernel_cache[(matrix_type, vessel.artery_id)] = (
                frequencies,
                kernel2,
                kernel_exact,
            )
            arrays[f"{matrix_type}__{vessel.artery_id}__kernel2"] = kernel2
            arrays[f"{matrix_type}__{vessel.artery_id}__kernel_exact"] = kernel_exact
            current_rows, current_exact, current_arrays = crossed_rows(
                vessel, matrix_type, eta, frequencies, kernel2, kernel_exact, config
            )
            matrix_rows.extend(current_rows)
            exact_rows.extend(current_exact)
            arrays.update(current_arrays)
            for waveform_index, row in enumerate(current_rows):
                matrix_values[matrix_type][vessel_index, waveform_index] = row["phi_rms"]
            if matrix_type == "physiological" and native_reference is not None:
                diagonal = next(
                    row for row in current_rows if row["waveform_id"] == vessel.artery_id
                )
                reference = float(native_reference.loc[vessel.artery_id, "phi_rms"])
                continuity_rows.append(
                    {
                        "artery_id": vessel.artery_id,
                        "step7_phi_rms": diagonal["phi_rms"],
                        "step6_phi_rms": reference,
                        "relative_error": abs(diagonal["phi_rms"] - reference)
                        / max(abs(reference), 1e-30),
                        "force_scale_n": force_scale(vessel),
                        "predicted_rms_excess_pn_at_epsilon_0p1": force_scale(vessel)
                        * 0.1**2
                        * diagonal["phi_rms"]
                        * 1e12,
                    }
                )

    matrices = pd.DataFrame(matrix_rows)
    exact = pd.DataFrame(exact_rows)
    continuity = pd.DataFrame(continuity_rows)
    decomposition_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for matrix_type, values in matrix_values.items():
        decomposition_rows.append(
            {"matrix_type": matrix_type, "scale": "raw", **additive_decomposition(values)}
        )
        decomposition_rows.append(
            {
                "matrix_type": matrix_type,
                "scale": "log",
                **additive_decomposition(np.log(values)),
            }
        )
        effect_rows.extend(matrix_effect_rows(matrix_type, values))
    decomposition = pd.DataFrame(decomposition_rows)
    effects = pd.DataFrame(effect_rows)

    rng = np.random.default_rng(config.random_seed)
    scrambles = [rng.uniform(-np.pi, np.pi, 6) for _ in range(config.phase_scrambles)]
    control_rows: list[dict[str, Any]] = []
    degeneracy_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for vessel in V2_ARTERY_CASES:
        frequencies, kernel2, _exact = kernel_cache[
            ("physiological", vessel.artery_id)
        ]
        rows, degeneracy = native_control_rows(
            vessel, frequencies, kernel2, config, scrambles
        )
        control_rows.extend(rows)
        degeneracy_rows.append(degeneracy)
        hydro_frequencies, hydro_kernel2, _hydro_exact = kernel_cache[
            ("hydrodynamic", vessel.artery_id)
        ]
        family_rows.extend(
            causal_family_rows(vessel, hydro_frequencies, hydro_kernel2, config)
        )
    controls = pd.DataFrame(control_rows)
    degeneracy = pd.DataFrame(degeneracy_rows)
    families = pd.DataFrame(family_rows)

    tables = {
        "crossed_susceptibility.csv": matrices,
        "crossed_exact_validation.csv": exact,
        "crossed_variance_decomposition.csv": decomposition,
        "crossed_main_effects.csv": effects,
        "step6_native_continuity.csv": continuity,
        "native_waveform_controls.csv": controls,
        "control_degeneracy_audit.csv": degeneracy,
        "causal_waveform_families.csv": families,
        "response_residuals.csv": pd.DataFrame(residual_rows),
    }
    for name, table in tables.items():
        table.to_csv(output_root / name, index=False)
    np.savez_compressed(output_root / "step7_archive.npz", **arrays)

    response_table = tables["response_residuals.csv"]
    gates: dict[str, Any] = {
        "step": 7,
        "profile": config.profile,
        "publication_profile": config.profile == "publication",
        "step6_gate_consumed": bool(step6.get("passed")),
        "two_matrices_36_entries_each": bool(
            (matrices.groupby("matrix_type").size() == 36).all()
        ),
        "native_diagonal_complete": int(matrices["native_diagonal"].sum()) == 12,
        "step6_native_continuity_passed": bool(
            continuity["relative_error"].max() <= config.closure_tolerance
        )
        if config.profile == "publication"
        else False,
        "all_crossed_exact_errors_below_1pct": bool(
            exact[
                ["waveform_relative_l2", "rms_relative_error", "spectrum_relative_l2"]
            ].to_numpy().max()
            <= config.exact_relative_limit
        ),
        "response_residual_passed": bool(
            response_table["max_normalized_response_residual"].max() <= 1e-10
        ),
        "variance_decomposition_closes": bool(
            np.allclose(
                decomposition[
                    ["vessel_fraction", "waveform_fraction", "interaction_fraction"]
                ].sum(axis=1),
                1.0,
                atol=1e-12,
            )
        ),
        "six_harmonic_ablations_per_artery": _control_count(
            controls, "harmonic_removal", 6
        ),
        "six_rms_matched_ablations_per_artery": _control_count(
            controls, "harmonic_removal_rms_matched", 6
        ),
        "phase_and_sign_controls_complete": _control_count(
            controls, "phase", config.phase_scrambles + 2
        )
        and _control_count(controls, "sign", 1),
        "sign_phase_degeneracy_audited": bool(
            degeneracy["sign_neutralized_equals_zero_phase_relative_error"].max()
            <= config.closure_tolerance
        ),
        "causal_families_complete": int(families["control"].nunique())
        == len(causal_waveform_families()),
        "low_rank_or_constitutive_robustness_run": False,
    }
    required = [
        name
        for name in gates
        if name not in {"step", "profile", "low_rank_or_constitutive_robustness_run"}
    ]
    gates["passed"] = all(bool(gates[name]) for name in required)
    (output_root / "step7_gate.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True), encoding="utf-8"
    )
    output_names = [*tables, "step7_archive.npz", "step7_gate.json"]
    manifest: dict[str, Any] = {
        "step": 7,
        "status": "complete" if gates["passed"] else "failed",
        "profile": config.profile,
        "step6_gate_consumed": bool(step6.get("passed")),
        "scientific_scope": "complete_waveform_experiments_and_crossed_matrices",
        "eta_reference": config.eta_reference,
        "exact_validation_epsilon": config.exact_epsilon,
        "allowed_next_step": 8 if gates["passed"] else None,
        "gates": gates,
        "files": {},
    }
    for name in output_names:
        path = output_root / name
        manifest["files"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (output_root / "step7_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "matrices": matrices,
        "exact": exact,
        "decomposition": decomposition,
        "effects": effects,
        "continuity": continuity,
        "controls": controls,
        "degeneracy": degeneracy,
        "families": families,
    }


def _control_count(controls: pd.DataFrame, family: str, expected: int) -> bool:
    return bool(
        (controls[controls["family"] == family].groupby("vessel_id").size() == expected).all()
    )
