from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES
from scipy.stats import spearmanr

from ._analysis_components import (
    _case_by_id,
    _fit_scale,
    _predict_scale,
    _rank_one,
    constitutive_robustness,
    crossed_matrices,
    harmonic_pair_attribution,
    native_atlas,
    waveform_catalogue,
    waveform_controls,
)
from .core import (
    AnalysisConfig,
    alpha_for_case,
    eta_for_case,
    force_scale,
    near_wall_basis,
    second_order_kernel,
    susceptibility_from_kernel,
    unit_perturbation_response,
)

_EPS = 1e-30
_DEFAULT_CONFIG = AnalysisConfig()


def build_operator_samples(
    config: AnalysisConfig = _DEFAULT_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Build dimensionless reciprocal interaction operators for all arteries."""

    config.validate()
    native_eta = np.asarray([eta_for_case(case) for case in V2_ARTERY_CASES])
    reference_eta = float(np.median(native_eta))
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for case in V2_ARTERY_CASES:
        unit = unit_perturbation_response(case, config)
        scale = force_scale(case)
        for condition, eta in (
            ("reference", reference_eta),
            ("physiological", eta_for_case(case)),
        ):
            near_wall = near_wall_basis(case, unit, config, eta)
            frequencies, dimensional_kernel = second_order_kernel(case, near_wall)
            dimensionless_kernel = dimensional_kernel / scale
            records.append(
                {
                    "condition": condition,
                    "artery_id": case.artery_id,
                    "artery_name": case.name,
                    "alpha": alpha_for_case(case),
                    "eta": eta,
                    "frequencies": frequencies,
                    "kernel": dimensional_kernel,
                    "dimensionless_kernel": dimensionless_kernel,
                    "kernel_norm": float(np.linalg.norm(dimensionless_kernel)),
                    "maximum_residual": unit.maximum_residual,
                }
            )
            prefix = f"{condition}__{case.artery_id}"
            arrays[f"{prefix}__frequencies"] = frequencies
            arrays[f"{prefix}__kernel"] = dimensional_kernel
            arrays[f"{prefix}__dimensionless_kernel"] = dimensionless_kernel
    return records, arrays


def _normalised_mean_kernel(records: Iterable[dict[str, Any]]) -> np.ndarray:
    kernels = [
        record["dimensionless_kernel"] / max(record["kernel_norm"], _EPS) for record in records
    ]
    if not kernels:
        raise ValueError("at least one operator sample is required")
    return np.mean(np.stack(kernels), axis=0)


def reduced_law_validation(
    records: Sequence[dict[str, Any]],
    config: AnalysisConfig = _DEFAULT_CONFIG,
    catalogue: Sequence[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    """Fit and cross-validate the dimensionless rank-one susceptibility law."""

    catalogue = waveform_catalogue() if catalogue is None else list(catalogue)
    predictions: list[dict[str, Any]] = []
    exponent_rows: list[dict[str, Any]] = []
    artery_ids = [case.artery_id for case in V2_ARTERY_CASES]

    for held_out in artery_ids:
        training = [record for record in records if record["artery_id"] != held_out]
        testing = [record for record in records if record["artery_id"] == held_out]
        parameters = _fit_scale(training)
        universal = _normalised_mean_kernel(training)
        reduced, _singular_values, retained = _rank_one(universal)
        exponent_rows.append(
            {
                "held_out_artery": held_out,
                "prefactor": float(np.exp(parameters[0])),
                "alpha_exponent": float(parameters[1]),
                "eta_exponent": float(parameters[2]),
                "retained_energy": retained,
            }
        )
        for record in testing:
            scale = _predict_scale(parameters, record["alpha"], record["eta"])
            case = _case_by_id(record["artery_id"])
            predicted_kernel = force_scale(case) * scale * reduced
            for waveform in catalogue:
                exact = susceptibility_from_kernel(
                    case,
                    record["frequencies"],
                    record["kernel"],
                    waveform["coefficients"],
                    config,
                ).rms
                predicted = susceptibility_from_kernel(
                    case,
                    record["frequencies"],
                    predicted_kernel,
                    waveform["coefficients"],
                    config,
                ).rms
                predictions.append(
                    {
                        "held_out_artery": held_out,
                        "condition": record["condition"],
                        "waveform_id": waveform["waveform_id"],
                        "family": waveform["family"],
                        "exact_phi_rms": exact,
                        "predicted_phi_rms": predicted,
                        "relative_error": abs(predicted - exact) / max(exact, _EPS),
                        "retained_energy": retained,
                    }
                )

    full_parameters = _fit_scale(records)
    universal = _normalised_mean_kernel(records)
    reduced, singular_values, retained = _rank_one(universal)
    prediction_frame = pd.DataFrame(predictions)
    native = prediction_frame[prediction_frame["family"] == "native"]
    correlations = [
        float(spearmanr(group["exact_phi_rms"], group["predicted_phi_rms"]).statistic)
        for _artery, group in native.groupby("held_out_artery")
    ]
    law = {
        "prefactor": float(np.exp(full_parameters[0])),
        "alpha_exponent": float(full_parameters[1]),
        "eta_exponent": float(full_parameters[2]),
        "retained_energy": retained,
        "median_relative_error": float(prediction_frame["relative_error"].median()),
        "p90_relative_error": float(prediction_frame["relative_error"].quantile(0.90)),
        "maximum_relative_error": float(prediction_frame["relative_error"].max()),
        "minimum_native_spearman": float(min(correlations)),
        "leave_one_out_exponents": exponent_rows,
    }
    arrays = {
        "universal_kernel": universal,
        "rank_one_kernel": reduced,
        "singular_values": singular_values,
        "scale_parameters": full_parameters,
    }
    return prediction_frame, law, arrays


def run_analysis(
    output_root: str | Path,
    config: AnalysisConfig = _DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Execute the complete public analysis and write reusable result files."""

    config.validate()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records, operator_arrays = build_operator_samples(config)
    atlas = native_atlas(records, config)
    crossed = crossed_matrices(records, config)
    controls = waveform_controls(records, config)
    pairs = harmonic_pair_attribution(records)
    predictions, law, reduction_arrays = reduced_law_validation(records, config)
    robustness = constitutive_robustness(config)

    tables = {
        "artery_atlas.csv": atlas,
        "crossed_susceptibility.csv": crossed,
        "waveform_controls.csv": controls,
        "harmonic_pair_attribution.csv": pairs,
        "reduced_law_validation.csv": predictions,
        "constitutive_robustness.csv": robustness,
    }
    for name, frame in tables.items():
        frame.to_csv(output_root / name, index=False)

    np.savez_compressed(output_root / "operator_archive.npz", **operator_arrays, **reduction_arrays)
    summary = {
        "software": "piconewton-waveform-susceptibility",
        "software_version": "1.0.1",
        "manuscript_title": (
            "Harmonic interactions shape anisotropy-induced transverse force in arterial blood flow"
        ),
        "configuration": asdict(config),
        "arteries": int(atlas["artery_id"].nunique()),
        "crossed_entries": int(len(crossed[crossed["condition"] == "physiological"])),
        "operator_samples": int(len(records)),
        "held_out_predictions": int(len(predictions)),
        "constitutive_paths": int(robustness["constitutive_path"].nunique()),
        "reduced_law": law,
        "scientific_scope": (
            "straight rigid axisymmetric six-harmonic anisotropic Womersley model"
        ),
        "claim_boundary": (
            "the reciprocal amplitude prefactor applies to beta=gamma and delta=1; "
            "other tensors require a separate constitutive amplitude factor"
        ),
    }
    (output_root / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "output_root": str(output_root),
        "atlas": atlas,
        "crossed": crossed,
        "controls": controls,
        "pairs": pairs,
        "predictions": predictions,
        "robustness": robustness,
        "summary": summary,
    }
