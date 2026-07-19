"""Corrected publication-facing study runner.

The solved Colab notebook remains untouched. This runner archives the physical
forcing and applies current-only primary decisions; calcium remains exploratory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .hypotheses import DecisionThresholds, classify_effects

PRIMARY_PATHWAYS = ("zero", "wss", "signed", "exposure", "vector")
CONTROL_PATHWAYS = (
    "exposure_work_matched", "exposure_peak_matched", "exposure_rms_matched",
    "signed_work_matched", "signed_peak_matched", "signed_rms_matched",
    "exposure_isotropic", "signed_isotropic", "exposure_elastic",
    "signed_elastic", "wss_elastic", "exposure_h2", "signed_h2",
    "wss_abs", "vector", "exposure", "signed", "zero", "wss",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def _require_files(root: Path, names: Iterable[str]) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required model outputs: {missing}")


def _load_waveforms(root: Path) -> dict[str, np.ndarray]:
    archive = np.load(root / "waveforms.npz")
    return {name: np.asarray(archive[name]) for name in archive.files}


def build_endpoint_decisions(
    effects: pd.DataFrame, thresholds: DecisionThresholds
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = classify_effects(effects, thresholds, primary_endpoint="current")
    calcium = classify_effects(effects, thresholds, primary_endpoint="calcium")
    calcium = calcium.rename(columns={"decision": "exploratory_calcium_signal"})
    return current, calcium


def build_directionality_audit(
    summary: pd.DataFrame, waveforms: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for artery in sorted(summary["artery_id"].unique()):
        current0 = waveforms[f"{artery}_zero_current_pA"]
        popen0 = waveforms[f"{artery}_zero_P_Open"]
        for pathway in PRIMARY_PATHWAYS[1:]:
            current = waveforms[f"{artery}_{pathway}_current_pA"]
            popen = waveforms[f"{artery}_{pathway}_P_Open"]
            delta_rms = _rms(current) - _rms(current0)
            rows.append({
                "artery_id": artery,
                "pathway": pathway,
                "reference": "zero_external_forcing",
                "delta_current_rms_pa": delta_rms,
                "delta_current_mean_abs_pa": float(np.mean(np.abs(current)) - np.mean(np.abs(current0))),
                "delta_signed_current_mean_pa": float(np.mean(current) - np.mean(current0)),
                "delta_popen_mean": float(np.mean(popen) - np.mean(popen0)),
                "response_direction": (
                    "unchanged" if abs(delta_rms) <= 1e-12 else
                    "increased_current_magnitude" if delta_rms > 0 else
                    "decreased_current_magnitude"
                ),
            })
    return pd.DataFrame(rows)


def build_domain_degeneracy_audit(
    summary: pd.DataFrame, waveforms: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for artery in sorted(summary["artery_id"].unique()):
        popen = float(np.max(np.abs(
            waveforms[f"{artery}_signed_P_Open"] - waveforms[f"{artery}_exposure_P_Open"]
        )))
        current = float(np.max(np.abs(
            waveforms[f"{artery}_signed_current_pA"] - waveforms[f"{artery}_exposure_current_pA"]
        )))
        calcium = float(np.max(np.abs(
            waveforms[f"{artery}_signed_calcium_nm"] - waveforms[f"{artery}_exposure_calcium_nm"]
        )))
        signed = summary[(summary.artery_id == artery) & (summary.pathway == "signed")].iloc[0]
        exposure = summary[(summary.artery_id == artery) & (summary.pathway == "exposure")].iloc[0]
        spatial = abs(float(signed.spatial_current_polarity_index) - float(exposure.spatial_current_polarity_index))
        degenerate = popen <= 1e-9 and current <= 1e-6 and calcium <= 1e-3
        rows.append({
            "artery_id": artery,
            "maximum_aggregate_popen_difference": popen,
            "maximum_aggregate_current_difference_pa": current,
            "maximum_aggregate_calcium_difference_nm": calcium,
            "spatial_polarity_index_difference": spatial,
            "aggregate_degenerate": degenerate,
            "interpretation": (
                "spatially_distinct_but_aggregate_indistinguishable"
                if degenerate and spatial > 0 else "aggregate_distinguishable"
            ),
        })
    return pd.DataFrame(rows)


def _split_pressure_key(prefix: str) -> tuple[str, str]:
    for pathway in CONTROL_PATHWAYS:
        marker = f"_{pathway}"
        if prefix.endswith(marker):
            return prefix[:-len(marker)], pathway
    raise ValueError(f"unrecognized pressure waveform key: {prefix}")


def build_pressure_clipping_audit(
    waveforms: dict[str, np.ndarray], *, maximum_pressure_mmhg: float
) -> pd.DataFrame:
    rows = []
    for name, values in sorted(waveforms.items()):
        domain = None
        for candidate in ("apical", "junctional"):
            suffix = f"_{candidate}_pressure_mmhg"
            if name.endswith(suffix):
                domain = candidate
                artery, pathway = _split_pressure_key(name[:-len(suffix)])
                break
        if domain is None:
            continue
        values = np.asarray(values, dtype=float)
        clipped = values >= maximum_pressure_mmhg - 1e-9
        rows.append({
            "artery_id": artery,
            "pathway": pathway,
            "domain": domain,
            "maximum_pressure_mmhg": float(np.max(values)),
            "clipped_fraction": float(np.mean(clipped)),
            "clipping_present": bool(np.any(clipped)),
            "pressure_ceiling_mmhg": maximum_pressure_mmhg,
        })
    return pd.DataFrame(rows)


def archive_hydrodynamic_inputs(
    *, package_root: Path, output_root: Path, profile: str,
    hydrodynamic_root: Path | None,
) -> dict[str, Any]:
    from .workflow import _load_hydrodynamic_items

    items, config, artifact = _load_hydrodynamic_items(
        package_root, profile, hydrodynamic_root
    )
    physical = {}
    direct = {}
    diagnostics = []
    direct_keys = (
        "time_cycle", "wss_anisotropic_pa", "wss_isotropic_pa",
        "force_signed_anisotropic_n", "force_signed_isotropic_n",
        "force_exposure_anisotropic_n", "force_exposure_isotropic_n",
    )
    extra_keys = (
        "time_s", "force_signed_anisotropy_increment_n",
        "force_exposure_anisotropy_increment_n",
    )
    for item in items:
        artery = str(item["artery_id"])
        for key in direct_keys:
            value = np.asarray(item[key])
            direct[f"{artery}_{key}"] = value
            physical[f"{artery}_{key}"] = value
        for key in extra_keys:
            if key in item:
                physical[f"{artery}_{key}"] = np.asarray(item[key])
        row = {"artery_id": artery}
        if isinstance(item.get("anisotropic_diagnostics"), dict):
            row.update(item["anisotropic_diagnostics"])
        diagnostics.append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_root / "physical_forcing_waveforms.npz", **physical)
    np.savez_compressed(output_root / "six_artery_hydrodynamics.npz", **direct)
    pd.DataFrame(diagnostics).to_csv(output_root / "hydrodynamic_diagnostics.csv", index=False)
    metadata = {
        "profile": profile,
        "hydrodynamic_config": asdict(config) if config is not None else None,
        "artifact": artifact,
        "array_count": len(physical),
        "arteries": sorted(str(item["artery_id"]) for item in items),
    }
    (output_root / "hydrodynamic_archive.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_completion_assessment(
    *, workflow_manifest: dict[str, Any], current_decisions: pd.DataFrame,
    degeneracy: pd.DataFrame, clipping: pd.DataFrame,
    hydrodynamic_archive: dict[str, Any],
) -> dict[str, Any]:
    passes = int((current_decisions.decision == "pass").sum())
    audit = workflow_manifest.get("calibration_audit", {})
    endpoint = workflow_manifest.get("endpoint_reference", {})
    calibration_complete = bool(audit.get("complete", False))
    endpoint_calibrated = endpoint.get("calibration_status") == "experimentally_calibrated"
    all_degenerate = bool(degeneracy.aggregate_degenerate.all())
    primary_clip = clipping[clipping.pathway.isin(PRIMARY_PATHWAYS)]
    clipping_absent = bool(primary_clip.empty or not primary_clip.clipping_present.any())
    structural = str(workflow_manifest.get("status", "")).startswith("passed")

    if not structural:
        outcome = "failed_numerical_or_structural_validation"
    elif passes == 0:
        outcome = "negative_under_current_parameterization"
    elif not calibration_complete or not endpoint_calibrated:
        outcome = "current_signal_requires_independent_calibration"
    elif all_degenerate:
        outcome = "aggregate_force_classes_not_identifiable"
    elif not clipping_absent:
        outcome = "candidate_signal_requires_pressure_ceiling_resolution"
    else:
        outcome = "candidate_for_independent_review"

    return {
        "assessment_version": "1.0",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "study_outcome": outcome,
        "claims_enabled": False,
        "primary_endpoint": "Piezo1 current RMS difference",
        "exploratory_endpoint": "calcium-scale proxy",
        "primary_hypothesis_pairs_passing": passes,
        "primary_hypothesis_pairs_total": int(len(current_decisions)),
        "gates": {
            "workflow_structural_validation": structural,
            "raw_hydrodynamic_forcing_archived": bool(hydrodynamic_archive.get("array_count", 0)),
            "current_endpoint_has_predeclared_cross_artery_support": passes > 0,
            "calibration_complete": calibration_complete,
            "endpoint_experimentally_calibrated": endpoint_calibrated,
            "signed_and_exposure_aggregate_outputs_distinguishable": not all_degenerate,
            "primary_pathways_free_of_pressure_clipping": clipping_absent,
            "independent_external_review_completed": False,
        },
        "claim_boundary": (
            "A positive biological claim requires current-based support, independent "
            "endpoint calibration, nondegenerate force-class observability, resolved "
            "pressure clipping, and independent review. Calcium cannot rescue a failed "
            "current decision."
        ),
    }


def run_scientific_study(
    *, package_root: Path, output_root: Path, profile: str = "full",
    calibration_path: Path | None = None,
    hydrodynamic_root: Path | None = None,
) -> dict[str, Any]:
    from .workflow import run_workflow

    package_root = Path(package_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    model_output = output_root / "model_outputs"
    assessment_output = output_root / "assessment"
    hydrodynamic_output = output_root / "hydrodynamics"
    assessment_output.mkdir(parents=True, exist_ok=False)

    hydrodynamic_archive = archive_hydrodynamic_inputs(
        package_root=package_root, output_root=hydrodynamic_output,
        profile=profile, hydrodynamic_root=hydrodynamic_root,
    )
    workflow_manifest = run_workflow(
        package_root=package_root, output_root=model_output, run_scan=False,
        profile=profile, hydrodynamic_root=hydrodynamic_output,
        calibration_path=calibration_path, require_calibrated=False,
    )
    _require_files(model_output, (
        "six_artery_summary.csv", "hypothesis_effects.csv", "waveforms.npz",
        "validation.json", "manifest.json",
    ))
    summary = pd.read_csv(model_output / "six_artery_summary.csv")
    effects = pd.read_csv(model_output / "hypothesis_effects.csv")
    waveforms = _load_waveforms(model_output)
    endpoint = workflow_manifest["endpoint_reference"]
    thresholds = DecisionThresholds(
        float(endpoint["current_detection_limit_pa"]),
        float(endpoint["calcium_detection_limit_nm"]), 4,
    )
    current, calcium = build_endpoint_decisions(effects, thresholds)
    current.to_csv(assessment_output / "primary_current_decisions.csv", index=False)
    calcium.to_csv(assessment_output / "exploratory_calcium_screen.csv", index=False)
    direction = build_directionality_audit(summary, waveforms)
    direction.to_csv(assessment_output / "primary_pathway_directionality.csv", index=False)
    degeneracy = build_domain_degeneracy_audit(summary, waveforms)
    degeneracy.to_csv(assessment_output / "signed_exposure_degeneracy_audit.csv", index=False)
    clipping = build_pressure_clipping_audit(
        waveforms,
        maximum_pressure_mmhg=float(workflow_manifest["interface_reference"]["maximum_pressure_mmhg"]),
    )
    clipping.to_csv(assessment_output / "pressure_clipping_audit.csv", index=False)
    assessment = build_completion_assessment(
        workflow_manifest=workflow_manifest, current_decisions=current,
        degeneracy=degeneracy, clipping=clipping,
        hydrodynamic_archive=hydrodynamic_archive,
    )
    (assessment_output / "scientific_assessment.json").write_text(
        json.dumps(assessment, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    produced = [path for path in output_root.rglob("*") if path.is_file()]
    manifest = {
        "workflow": "picoNewton_v4_scientific_study",
        "status": "completed_with_claims_disabled",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "study_outcome": assessment["study_outcome"],
        "notebook_modified": False,
        "claims_enabled": False,
        "outputs": {
            str(path.relative_to(output_root)): {
                "bytes": path.stat().st_size, "sha256": _sha256(path)
            }
            for path in sorted(produced)
        },
    }
    (output_root / "scientific_study_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the corrected current-primary picoNewton_v4 study."
    )
    parser.add_argument("--package-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--hydrodynamic-root", type=Path)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    if not (root / "pyproject.toml").exists() and (root / "picoNewton_v4").exists():
        root = root / "picoNewton_v4"
    manifest = run_scientific_study(
        package_root=root, output_root=args.output, profile=args.profile,
        calibration_path=args.calibration,
        hydrodynamic_root=args.hydrodynamic_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0 if manifest["status"] == "completed_with_claims_disabled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
