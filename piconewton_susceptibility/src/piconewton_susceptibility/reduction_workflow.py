from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES
from scipy.stats import spearmanr

from .reduction_core import (
    Step8Config,
    fit_power_law,
    fit_scalar_moment,
    kernel_scale,
    predict_power_law,
    predict_scalar_moment,
    susceptibility_from_kernel,
    truncated_kernel,
    universal_kernel,
    waveform_catalog,
)
from .reduction_support import file_records, kernel_key, summarise_errors, validate_step7


def _records(crossed: pd.DataFrame, archive: Any) -> list[dict[str, Any]]:
    meta = crossed[["matrix_type", "vessel_id", "vessel_name", "alpha", "eta"]]
    rows = []
    for item in meta.drop_duplicates().itertuples(index=False):
        kernel = np.asarray(archive[kernel_key(archive, item.matrix_type, item.vessel_id)])
        rows.append(
            {
                "matrix_type": item.matrix_type,
                "vessel_id": item.vessel_id,
                "vessel_name": item.vessel_name,
                "alpha": float(item.alpha),
                "eta": float(item.eta),
                "kernel": kernel,
                "kernel_scale": kernel_scale(kernel),
            }
        )
    return rows


def _scale_fit(rows: list[dict[str, Any]]) -> np.ndarray:
    return fit_power_law(
        np.array([row["alpha"] for row in rows]),
        np.array([row["eta"] for row in rows]),
        np.array([row["kernel_scale"] for row in rows]),
    )


def _rankings(predictions: pd.DataFrame, rank: int) -> pd.DataFrame:
    selected = predictions.query("rank == @rank and family == 'native'")
    rows = []
    for matrix_type, current in selected.groupby("matrix_type"):
        for level, group in current.groupby("held_out_artery"):
            rows.append(
                {
                    "matrix_type": matrix_type,
                    "ranking_axis": "waveforms_within_held_vessel",
                    "level_id": level,
                    "spearman": float(
                        spearmanr(group["exact_phi_rms"], group["predicted_phi_rms"]).statistic
                    ),
                }
            )
        for level, group in current.groupby("waveform_id"):
            rows.append(
                {
                    "matrix_type": matrix_type,
                    "ranking_axis": "held_vessels_for_waveform",
                    "level_id": level,
                    "spearman": float(
                        spearmanr(group["exact_phi_rms"], group["predicted_phi_rms"]).statistic
                    ),
                }
            )
    return pd.DataFrame(rows)


def _modal_cross_validation(records, catalogue, config):
    predictions, spectra, exponents = [], [], []
    artery_ids = [case.artery_id for case in V2_ARTERY_CASES]
    for held in artery_ids:
        train = [row for row in records if row["vessel_id"] != held]
        test = [row for row in records if row["vessel_id"] == held]
        parameters = _scale_fit(train)
        mean_kernel = universal_kernel(row["kernel"] for row in train)
        exponents.append(
            {
                "held_out_artery": held,
                "log_prefactor": float(parameters[0]),
                "prefactor": float(np.exp(parameters[0])),
                "alpha_exponent": float(parameters[1]),
                "eta_exponent": float(parameters[2]),
            }
        )
        for rank in config.candidate_ranks:
            reduced, singular_values, retained = truncated_kernel(mean_kernel, rank)
            spectra.extend(
                {
                    "held_out_artery": held,
                    "candidate_rank": rank,
                    "singular_index": index,
                    "singular_value": float(value),
                    "retained_energy": retained,
                }
                for index, value in enumerate(singular_values, start=1)
            )
            for row in test:
                scale = predict_power_law(parameters, row["alpha"], row["eta"])
                scale_error = abs(scale - row["kernel_scale"]) / row["kernel_scale"]
                for waveform in catalogue:
                    coefficients = waveform["coefficients"]
                    exact = susceptibility_from_kernel(row["kernel"], coefficients)
                    predicted = scale * susceptibility_from_kernel(reduced, coefficients)
                    predictions.append(
                        {
                            "held_out_artery": held,
                            "matrix_type": row["matrix_type"],
                            "waveform_id": waveform["waveform_id"],
                            "family": waveform["family"],
                            "source_artery": waveform["source_artery"],
                            "rank": rank,
                            "retained_kernel_energy": retained,
                            "exact_phi_rms": exact,
                            "predicted_phi_rms": predicted,
                            "relative_error": abs(predicted - exact) / max(exact, 1e-30),
                            "scale_relative_error": scale_error,
                        }
                    )
    return pd.DataFrame(predictions), pd.DataFrame(spectra), pd.DataFrame(exponents)


def _scalar_cross_validation(records, catalogue):
    rows = []
    families = sorted({waveform["family"] for waveform in catalogue})
    for held in [case.artery_id for case in V2_ARTERY_CASES]:
        train = [row for row in records if row["vessel_id"] != held]
        test = [row for row in records if row["vessel_id"] == held]
        for family in families:
            fitting = []
            for row in train:
                for waveform in catalogue:
                    if waveform["family"] != family:
                        coefficients = waveform["coefficients"]
                        fitting.append(
                            (
                                row["alpha"],
                                row["eta"],
                                coefficients,
                                susceptibility_from_kernel(row["kernel"], coefficients),
                            )
                        )
            parameters, exponent = fit_scalar_moment(fitting)
            for row in test:
                for waveform in catalogue:
                    if waveform["family"] == family:
                        exact = susceptibility_from_kernel(
                            row["kernel"], waveform["coefficients"]
                        )
                        predicted = predict_scalar_moment(
                            parameters,
                            exponent,
                            row["alpha"],
                            row["eta"],
                            waveform["coefficients"],
                        )
                        rows.append(
                            {
                                "held_out_artery": held,
                                "held_out_family": family,
                                "matrix_type": row["matrix_type"],
                                "waveform_id": waveform["waveform_id"],
                                "exact_phi_rms": exact,
                                "predicted_phi_rms": predicted,
                                "relative_error": abs(predicted - exact) / max(exact, 1e-30),
                                "moment_exponent": exponent,
                                "alpha_exponent": float(parameters[1]),
                                "eta_exponent": float(parameters[2]),
                            }
                        )
    return pd.DataFrame(rows)


def _selection(predictions, scalar, config):
    ranks = summarise_errors(predictions, ["rank"])
    families = summarise_errors(predictions, ["rank", "family"])
    scalar_summary = summarise_errors(scalar, ["held_out_family"])
    rows = []
    for rank in config.candidate_ranks:
        overall = ranks[ranks["rank"] == rank].iloc[0]
        current = families[families["rank"] == rank]
        retained = predictions.query("rank == @rank")["retained_kernel_energy"].min()
        passed = bool(
            retained >= config.retained_energy_min
            and overall["median_relative_error"] <= config.median_relative_error_max
            and overall["p90_relative_error"] <= config.p90_relative_error_max
            and overall["maximum_relative_error"] <= config.maximum_relative_error_max
            and current["median_relative_error"].max()
            <= config.family_median_relative_error_max
            and current["maximum_relative_error"].max()
            <= config.family_maximum_relative_error_max
        )
        rows.append(
            {
                "candidate": f"rank_{rank}_universal_kernel",
                "rank": rank,
                "retained_energy_min": retained,
                "median_relative_error": overall["median_relative_error"],
                "p90_relative_error": overall["p90_relative_error"],
                "maximum_relative_error": overall["maximum_relative_error"],
                "maximum_family_median_error": current["median_relative_error"].max(),
                "maximum_family_error": current["maximum_relative_error"].max(),
                "passed": passed,
            }
        )
    scalar_passed = bool(
        scalar_summary["median_relative_error"].max()
        <= config.family_median_relative_error_max
        and scalar_summary["maximum_relative_error"].max()
        <= config.family_maximum_relative_error_max
    )
    rows.append(
        {
            "candidate": "inverse_harmonic_scalar_moment",
            "rank": 0,
            "retained_energy_min": np.nan,
            "median_relative_error": scalar["relative_error"].median(),
            "p90_relative_error": scalar["relative_error"].quantile(0.90),
            "maximum_relative_error": scalar["relative_error"].max(),
            "maximum_family_median_error": scalar_summary["median_relative_error"].max(),
            "maximum_family_error": scalar_summary["maximum_relative_error"].max(),
            "passed": scalar_passed,
        }
    )
    return ranks, families, scalar_summary, pd.DataFrame(rows)


def run_reduction_study(output_root, step7_root, config=None):
    config = config or Step8Config()
    config.validate()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    step7 = validate_step7(step7_root)
    step7_root = Path(step7["root"])
    crossed = pd.read_csv(step7_root / "crossed_susceptibility.csv")
    archive = np.load(step7_root / "step7_archive.npz")
    records = _records(crossed, archive)
    if len(records) != 12:
        raise RuntimeError(f"Step 8 requires twelve operator samples, found {len(records)}")
    catalogue = waveform_catalog(config)
    predictions, singular, exponents = _modal_cross_validation(records, catalogue, config)
    scalar = _scalar_cross_validation(records, catalogue)
    rank_summary, family_summary, scalar_summary, selection = _selection(
        predictions, scalar, config
    )
    passing = selection.query("rank >= 1 and passed").sort_values("rank")
    selected_rank = int(passing.iloc[0]["rank"]) if not passing.empty else None
    ranking = _rankings(predictions, selected_rank) if selected_rank else pd.DataFrame()

    parameters = _scale_fit(records)
    full_kernel = universal_kernel(row["kernel"] for row in records)
    selected_kernel, singular_values, retained = truncated_kernel(full_kernel, selected_rank or 1)
    exponent_span = max(
        exponents["alpha_exponent"].max() - exponents["alpha_exponent"].min(),
        exponents["eta_exponent"].max() - exponents["eta_exponent"].min(),
    )
    law = {
        "law": "Phi_hat = C * alpha^p_alpha * eta^p_eta * Psi_R(g)",
        "waveform_functional": (
            "Psi_R(g) is the Parseval RMS generated by the rank-R universal "
            "two-sided interaction kernel"
        ),
        "selected_rank": selected_rank,
        "prefactor": float(np.exp(parameters[0])),
        "alpha_exponent": float(parameters[1]),
        "eta_exponent": float(parameters[2]),
        "retained_kernel_energy": retained,
        "scalar_moment_selected": False,
        "claim_boundary": (
            "valid within the straight rigid reciprocal anisotropic Womersley model, "
            "six-harmonic waveform class and tested alpha-eta domain"
        ),
    }
    np.savez_compressed(
        output_root / "step8_reduced_law.npz",
        universal_kernel=full_kernel,
        selected_kernel=selected_kernel,
        singular_values=singular_values,
        scale_parameters=parameters,
    )
    (output_root / "reduced_law.json").write_text(
        json.dumps(law, indent=2, sort_keys=True), encoding="utf-8"
    )
    tables = {
        "kernel_mode_spectrum.csv": singular,
        "vessel_scaling_loso.csv": exponents,
        "compact_law_predictions.csv": predictions,
        "compact_law_rank_summary.csv": rank_summary,
        "compact_law_family_summary.csv": family_summary,
        "scalar_moment_double_holdout.csv": scalar,
        "scalar_moment_summary.csv": scalar_summary,
        "model_selection.csv": selection,
        "native_ranking_validation.csv": ranking,
    }
    for name, frame in tables.items():
        frame.to_csv(output_root / name, index=False)

    selected = selection[selection["rank"] == selected_rank].iloc[0] if selected_rank else None
    scalar_passed = bool(selection.query("rank == 0").iloc[0]["passed"])
    gates = {
        "step": 8,
        "profile": config.profile,
        "step7_gate_consumed": True,
        "twelve_operator_samples": len(records) == 12,
        "three_candidate_ranks_evaluated": set(selection.query("rank >= 1")["rank"])
        == set(config.candidate_ranks),
        "rank_one_selected": selected_rank == 1,
        "selected_model_cross_validation_passed": bool(selected["passed"])
        if selected is not None
        else False,
        "ranking_preserved": bool(
            not ranking.empty and ranking["spearman"].min() >= config.ranking_spearman_min
        ),
        "vessel_exponents_stable": bool(exponent_span <= config.exponent_span_max),
        "scalar_candidate_evaluated": len(scalar) > 0,
        "scalar_candidate_rejected_if_nonuniversal": not scalar_passed,
        "constitutive_robustness_run": False,
        "biological_endpoint_model_run": False,
    }
    excluded = {"step", "profile", "constitutive_robustness_run", "biological_endpoint_model_run"}
    gates["passed"] = all(bool(value) for name, value in gates.items() if name not in excluded)
    (output_root / "step8_gate.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True), encoding="utf-8"
    )
    output_names = [*tables, "step8_reduced_law.npz", "reduced_law.json", "step8_gate.json"]
    manifest = {
        "step": 8,
        "status": "complete" if gates["passed"] else "failed",
        "profile": config.profile,
        "scientific_scope": "general_reduced_law_and_held_out_validation",
        "selected_rank": selected_rank,
        "allowed_next_step": 9 if gates["passed"] else None,
        "gates": gates,
        "files": file_records(output_root, output_names),
    }
    (output_root / "step8_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "law": law,
        "predictions": predictions,
        "rank_summary": rank_summary,
        "family_summary": family_summary,
        "selection": selection,
        "scalar_summary": scalar_summary,
        "ranking": ranking,
        "exponents": exponents,
    }
