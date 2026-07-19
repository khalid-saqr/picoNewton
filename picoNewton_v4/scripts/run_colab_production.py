#!/usr/bin/env python3
"""Full-resolution parametric picoNewton_v4 study and Nature-style figures."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from piconewton_v4.calibration import load_parameterization
from piconewton_v4.scientific_study import run_scientific_study
from piconewton_v4.workflow import _load_hydrodynamic_items, _pathway, _simulate

CURRENT_THRESHOLD_PA = 3.3
MINIMUM_PASSING_ARTERIES = 4
EXPECTED_ARTERIES = {
    "aortic_root", "thoracic_aorta", "femoral", "carotid", "iliac", "brachial"
}
ARTERY_ORDER = [
    "aortic_root", "thoracic_aorta", "carotid", "femoral", "iliac", "brachial"
]
ARTERY_LABELS = {
    "aortic_root": "Aortic root",
    "thoracic_aorta": "Thoracic aorta",
    "carotid": "Carotid",
    "femoral": "Femoral",
    "iliac": "Iliac",
    "brachial": "Brachial",
}
PALETTE = {
    "black": "#000000",
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#666666",
}
PATHWAY_COLORS = {
    "zero": PALETTE["grey"],
    "wss": PALETTE["black"],
    "signed": PALETTE["blue"],
    "exposure": PALETTE["orange"],
    "vector": PALETTE["green"],
}
MM_TO_INCH = 1.0 / 25.4
NATURE_DOUBLE_COLUMN_MM = 183.0
NATURE_MAX_HEIGHT_MM = 170.0


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def _rms_difference(result_a: dict[str, Any], result_b: dict[str, Any]) -> float:
    a = np.asarray(result_a["aggregate"]["current_pA"], dtype=float)
    b = np.asarray(result_b["aggregate"]["current_pA"], dtype=float)
    return _rms(a - b)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parameter_scenarios() -> pd.DataFrame:
    base = {
        "localization_area_um2": 10.0,
        "force_transfer_fraction": 1.0,
        "channel_count": 4165.0,
        "apical_channel_fraction": 0.5,
        "fast_fraction": 0.5,
        "maximum_pressure_mmhg": 70.0,
    }
    specs = [
        ("baseline", "baseline", base.copy()),
        ("area_0p5", "localization_area_um2", {**base, "localization_area_um2": 0.5}),
        ("area_3", "localization_area_um2", {**base, "localization_area_um2": 3.0}),
        ("transfer_0p1", "force_transfer_fraction", {**base, "force_transfer_fraction": 0.1}),
        ("transfer_0p3", "force_transfer_fraction", {**base, "force_transfer_fraction": 0.3}),
        ("channels_1000", "channel_count", {**base, "channel_count": 1000.0}),
        ("channels_10000", "channel_count", {**base, "channel_count": 10000.0}),
        ("apical_0p25", "apical_channel_fraction", {**base, "apical_channel_fraction": 0.25}),
        ("apical_0p75", "apical_channel_fraction", {**base, "apical_channel_fraction": 0.75}),
        ("fast_0p25", "fast_fraction", {**base, "fast_fraction": 0.25}),
        ("fast_0p75", "fast_fraction", {**base, "fast_fraction": 0.75}),
        ("ceiling_35", "maximum_pressure_mmhg", {**base, "maximum_pressure_mmhg": 35.0}),
        ("ceiling_140", "maximum_pressure_mmhg", {**base, "maximum_pressure_mmhg": 140.0}),
    ]
    rows = []
    for scenario_id, varied_parameter, values in specs:
        rows.append({
            "scenario_id": scenario_id,
            "varied_parameter": varied_parameter,
            "varied_value": np.nan if varied_parameter == "baseline" else values[varied_parameter],
            **values,
        })
    return pd.DataFrame(rows)


def run_parameter_ensemble(
    *, package_root: Path, study_root: Path, output_root: Path, profile: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_root.mkdir(parents=True, exist_ok=False)
    scenarios = parameter_scenarios()
    scenarios.to_csv(output_root / "parameter_scenarios.csv", index=False)
    calibration_path = package_root / "configs" / "literature_calibration.json"
    base_interface, base_endpoint, _ = load_parameterization(calibration_path)
    hydro_items, _, _ = _load_hydrodynamic_items(
        package_root, profile, study_root / "hydrodynamics"
    )
    rows: list[dict[str, Any]] = []
    waveforms: dict[str, np.ndarray] = {}

    for index, scenario in scenarios.iterrows():
        scenario_id = str(scenario["scenario_id"])
        print(f"Parameter scenario {index + 1:02d}/{len(scenarios):02d}: {scenario_id}", flush=True)
        fast_fraction = float(scenario["fast_fraction"])
        interface = replace(
            base_interface,
            normal=replace(base_interface.normal, fast_fraction=fast_fraction),
            tangential=replace(base_interface.tangential, fast_fraction=fast_fraction),
            signed_force_area_m2=float(scenario["localization_area_um2"]) * 1e-12,
            exposure_area_m2=float(scenario["localization_area_um2"]) * 1e-12,
            signed_force_transfer_fraction=float(scenario["force_transfer_fraction"]),
            exposure_transfer_fraction=float(scenario["force_transfer_fraction"]),
            apical_channel_fraction=float(scenario["apical_channel_fraction"]),
            maximum_pressure_mmhg=float(scenario["maximum_pressure_mmhg"]),
        )
        endpoint = replace(base_endpoint, channel_count=float(scenario["channel_count"]))

        for item in hydro_items:
            artery = str(item["artery_id"])
            time_s = np.asarray(item["time_s"], dtype=float)
            dt_s = float(time_s[1] - time_s[0])
            zero_force = np.zeros_like(np.asarray(item["force_signed_anisotropic_n"], dtype=float))
            responses = {
                pathway: _pathway(item, pathway, interface=interface, endpoint=endpoint)
                for pathway in ("zero", "wss", "signed", "exposure", "vector")
            }
            responses["wss_abs"] = _simulate(
                wss_pa=np.abs(np.asarray(item["wss_anisotropic_pa"], dtype=float)),
                signed_force_n=zero_force,
                exposure_force_n=zero_force,
                dt_s=dt_s,
                interface=interface,
                endpoint=endpoint,
            )
            signed_h3 = _rms_difference(responses["signed"], responses["wss"])
            exposure_h3 = _rms_difference(responses["exposure"], responses["wss_abs"])
            zero_rms = _rms(responses["zero"]["aggregate"]["current_pA"])
            clipping_fractions = []
            for pathway in ("wss", "signed", "exposure", "vector"):
                membrane = responses[pathway]["membrane"]
                for domain in ("apical_pressure_mmhg", "junctional_pressure_mmhg"):
                    pressure = np.asarray(membrane[domain], dtype=float)
                    clipping_fractions.append(float(np.mean(
                        pressure >= float(scenario["maximum_pressure_mmhg"]) - 1e-9
                    )))
                waveforms[f"{scenario_id}__{artery}__{pathway}__current_pA"] = np.asarray(
                    responses[pathway]["aggregate"]["current_pA"], dtype=float
                )
            waveforms[f"{artery}__time_s"] = time_s
            rows.append({
                "scenario_id": scenario_id,
                "varied_parameter": scenario["varied_parameter"],
                "varied_value": scenario["varied_value"],
                "artery_id": artery,
                "signed_vs_wss_current_rms_difference_pa": signed_h3,
                "exposure_vs_abs_wss_current_rms_difference_pa": exposure_h3,
                "signed_passes_current_threshold": signed_h3 >= CURRENT_THRESHOLD_PA,
                "exposure_passes_current_threshold": exposure_h3 >= CURRENT_THRESHOLD_PA,
                "signed_delta_current_rms_pa": _rms(responses["signed"]["aggregate"]["current_pA"]) - zero_rms,
                "exposure_delta_current_rms_pa": _rms(responses["exposure"]["aggregate"]["current_pA"]) - zero_rms,
                "vector_delta_current_rms_pa": _rms(responses["vector"]["aggregate"]["current_pA"]) - zero_rms,
                "signed_exposure_current_rms_difference_pa": _rms_difference(
                    responses["signed"], responses["exposure"]
                ),
                "maximum_primary_clipped_fraction": max(clipping_fractions),
            })
        gc.collect()

    results = pd.DataFrame(rows)
    results.to_csv(output_root / "parametric_artery_results.csv", index=False)
    np.savez_compressed(output_root / "parametric_current_waveforms.npz", **waveforms)
    summary = (
        results.groupby(["scenario_id", "varied_parameter", "varied_value"], dropna=False)
        .agg(
            signed_passing_arteries=("signed_passes_current_threshold", "sum"),
            exposure_passing_arteries=("exposure_passes_current_threshold", "sum"),
            signed_median_h3_current_difference_pa=("signed_vs_wss_current_rms_difference_pa", "median"),
            exposure_median_h3_current_difference_pa=("exposure_vs_abs_wss_current_rms_difference_pa", "median"),
            signed_maximum_h3_current_difference_pa=("signed_vs_wss_current_rms_difference_pa", "max"),
            exposure_maximum_h3_current_difference_pa=("exposure_vs_abs_wss_current_rms_difference_pa", "max"),
            maximum_primary_clipped_fraction=("maximum_primary_clipped_fraction", "max"),
            maximum_signed_exposure_current_difference_pa=("signed_exposure_current_rms_difference_pa", "max"),
        )
        .reset_index()
        .merge(scenarios, on=["scenario_id", "varied_parameter", "varied_value"], how="left")
    )
    summary["signed_cross_artery_pass"] = summary["signed_passing_arteries"] >= MINIMUM_PASSING_ARTERIES
    summary["exposure_cross_artery_pass"] = summary["exposure_passing_arteries"] >= MINIMUM_PASSING_ARTERIES
    summary.to_csv(output_root / "parameter_scenario_summary.csv", index=False)
    return results, summary


def configure_figure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 7,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _panel_label(ax: Any, letter: str) -> None:
    ax.text(-0.16, 1.08, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left", color="black")


def _clean_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def _save_figure(fig: Any, figure_root: Path, stem: str) -> list[Path]:
    produced = []
    for extension, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        path = figure_root / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **kwargs)
        produced.append(path)
    return produced


def generate_nature_figures(
    *, study_root: Path, parameter_root: Path, figure_root: Path
) -> list[Path]:
    figure_root.mkdir(parents=True, exist_ok=False)
    configure_figure_style()
    summary = pd.read_csv(study_root / "model_outputs" / "six_artery_summary.csv")
    effects = pd.read_csv(study_root / "model_outputs" / "hypothesis_effects.csv")
    decisions = pd.read_csv(study_root / "assessment" / "primary_current_decisions.csv")
    directionality = pd.read_csv(study_root / "assessment" / "primary_pathway_directionality.csv")
    degeneracy = pd.read_csv(study_root / "assessment" / "signed_exposure_degeneracy_audit.csv")
    parameter_summary = pd.read_csv(parameter_root / "parameter_scenario_summary.csv")
    physical = np.load(study_root / "hydrodynamics" / "physical_forcing_waveforms.npz")
    produced: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(NATURE_DOUBLE_COLUMN_MM * MM_TO_INCH, 142 * MM_TO_INCH))
    axes = axes.ravel()
    for artery in ARTERY_ORDER:
        phase = np.asarray(physical[f"{artery}_time_cycle"], dtype=float)
        axes[0].plot(phase, physical[f"{artery}_wss_anisotropic_pa"], label=ARTERY_LABELS[artery])
        axes[1].plot(phase, 1e12 * physical[f"{artery}_force_signed_anisotropic_n"])
        axes[2].plot(phase, 1e12 * physical[f"{artery}_force_exposure_anisotropic_n"])
    axes[0].set(xlabel="Cardiac-cycle phase", ylabel="Wall shear stress (Pa)", title="Anisotropic wall shear stress")
    axes[1].set(xlabel="Cardiac-cycle phase", ylabel="Signed force (pN)", title="Signed transverse force")
    axes[2].set(xlabel="Cardiac-cycle phase", ylabel="Exposure (pN)", title="Nonnegative force exposure")
    baseline = summary[summary["pathway"].isin(["zero", "wss", "signed", "exposure", "vector"])]
    for pathway in ("zero", "wss", "signed", "exposure", "vector"):
        values = baseline[baseline["pathway"] == pathway].set_index("artery_id").reindex(ARTERY_ORDER)
        axes[3].plot(np.arange(6), values["current_rms_pa"], marker="o",
                     color=PATHWAY_COLORS[pathway], label=pathway)
    axes[3].set_xticks(np.arange(6), [ARTERY_LABELS[a] for a in ARTERY_ORDER], rotation=35, ha="right")
    axes[3].set(ylabel="RMS Piezo1 current (pA)", title="Baseline pathway responses")
    axes[0].legend(frameon=False, ncol=2)
    axes[3].legend(frameon=False, ncol=3)
    for letter, ax in zip("abcd", axes):
        _panel_label(ax, letter); _clean_axes(ax)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.11, top=0.95, wspace=0.34, hspace=0.42)
    produced.extend(_save_figure(fig, figure_root, "Figure_1_forcing_and_baseline_response"))
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(NATURE_DOUBLE_COLUMN_MM * MM_TO_INCH, 150 * MM_TO_INCH))
    axes = axes.ravel()
    h3 = effects[effects["hypothesis"] == "H3a"]
    for target, marker, color in (("signed", "o", PALETTE["blue"]), ("exposure", "s", PALETTE["orange"])):
        selected = h3[h3["target"] == target].set_index("artery_id").reindex(ARTERY_ORDER)
        axes[0].plot(np.arange(6), selected["current_rms_difference_pa"], marker=marker, color=color, label=target)
    axes[0].axhline(CURRENT_THRESHOLD_PA, color=PALETTE["vermillion"], linestyle="--")
    axes[0].set_xticks(np.arange(6), [ARTERY_LABELS[a] for a in ARTERY_ORDER], rotation=35, ha="right")
    axes[0].set(ylabel="RMS current difference (pA)", title="Direct force versus WSS")
    axes[0].legend(frameon=False)
    decision_labels = decisions["hypothesis"].astype(str) + " | " + decisions["target"].astype(str)
    order = np.argsort(decisions["passing_arteries"].to_numpy())
    axes[1].barh(np.arange(len(order)), decisions.iloc[order]["passing_arteries"], color=PALETTE["blue"])
    axes[1].axvline(MINIMUM_PASSING_ARTERIES, color=PALETTE["vermillion"], linestyle="--")
    axes[1].set_yticks(np.arange(len(order)), decision_labels.iloc[order])
    axes[1].set(xlabel="Arteries exceeding current threshold", title="Current-primary decisions", xlim=(0, 6.2))
    matrix = directionality.pivot(index="pathway", columns="artery_id", values="delta_current_rms_pa").reindex(
        index=["wss", "signed", "exposure", "vector"], columns=ARTERY_ORDER
    )
    vmax = float(np.nanmax(np.abs(matrix.to_numpy())))
    image = axes[2].imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[2].set_xticks(np.arange(6), [ARTERY_LABELS[a] for a in ARTERY_ORDER], rotation=35, ha="right")
    axes[2].set_yticks(np.arange(4), matrix.index)
    axes[2].set_title("Change from zero external forcing")
    colorbar = fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    colorbar.set_label("Δ RMS current (pA)", fontsize=6)
    colorbar.ax.tick_params(labelsize=5)
    deg = degeneracy.set_index("artery_id").reindex(ARTERY_ORDER)
    axes[3].bar(np.arange(6), np.maximum(deg["maximum_aggregate_current_difference_pa"], 1e-12), color=PALETTE["purple"])
    axes[3].axhline(1e-6, color=PALETTE["vermillion"], linestyle="--")
    axes[3].set_yscale("log")
    axes[3].set_xticks(np.arange(6), [ARTERY_LABELS[a] for a in ARTERY_ORDER], rotation=35, ha="right")
    axes[3].set(ylabel="Maximum current difference (pA)", title="Signed–exposure aggregate degeneracy")
    for letter, ax in zip("abcd", axes):
        _panel_label(ax, letter)
        if ax is not axes[2]: _clean_axes(ax)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.95, wspace=0.42, hspace=0.48)
    produced.extend(_save_figure(fig, figure_root, "Figure_2_current_primary_assessment"))
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(NATURE_DOUBLE_COLUMN_MM * MM_TO_INCH, 148 * MM_TO_INCH))
    axes = axes.ravel()
    panels = [
        ("localization_area_um2", "Localization area (µm²)", "log"),
        ("force_transfer_fraction", "Force-transfer fraction", "linear"),
        ("channel_count", "Channel count", "log"),
        ("apical_channel_fraction", "Apical channel fraction", "linear"),
        ("fast_fraction", "Fast viscoelastic fraction", "linear"),
    ]
    baseline_row = parameter_summary[parameter_summary["scenario_id"] == "baseline"]
    for ax, (parameter, xlabel, scale) in zip(axes[:5], panels):
        selected = pd.concat([baseline_row, parameter_summary[parameter_summary["varied_parameter"] == parameter]])
        selected = selected.drop_duplicates("scenario_id").sort_values(parameter)
        ax.plot(selected[parameter], selected["signed_median_h3_current_difference_pa"], marker="o", color=PALETTE["blue"], label="signed")
        ax.plot(selected[parameter], selected["exposure_median_h3_current_difference_pa"], marker="s", color=PALETTE["orange"], label="exposure")
        ax.axhline(CURRENT_THRESHOLD_PA, color=PALETTE["vermillion"], linestyle="--")
        if scale == "log": ax.set_xscale("log")
        ax.set(xlabel=xlabel, ylabel="Median H3 current difference (pA)")
        _clean_axes(ax)
    pressure = pd.concat([baseline_row, parameter_summary[parameter_summary["varied_parameter"] == "maximum_pressure_mmhg"]])
    pressure = pressure.drop_duplicates("scenario_id").sort_values("maximum_pressure_mmhg")
    axes[5].plot(pressure["maximum_pressure_mmhg"], pressure["maximum_primary_clipped_fraction"], marker="o", color=PALETTE["vermillion"])
    axes[5].set(xlabel="Pressure ceiling (mmHg)", ylabel="Maximum clipped cycle fraction", ylim=(-0.02, 1.02))
    _clean_axes(axes[5])
    axes[0].legend(frameon=False)
    titles = ["Localization", "Force transfer", "Channel abundance", "Spatial allocation", "Viscoelastic partition", "Pressure clipping"]
    for letter, title, ax in zip("abcdef", titles, axes):
        ax.set_title(title); _panel_label(ax, letter)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.11, top=0.94, wspace=0.42, hspace=0.48)
    produced.extend(_save_figure(fig, figure_root, "Figure_3_parametric_robustness"))
    plt.close(fig)

    legends = """Figure 1 | Full-resolution arterial forcing and baseline mechanosensory response. a, Anisotropic wall shear stress across the six arteries. b, Signed transverse Lamb force. c, Nonnegative force-exposure signal. d, RMS Piezo1 current for the zero-input, WSS, signed-force, exposure, and combined-vector pathways under the literature-constrained reference parameterization.\n\nFigure 2 | Current-primary hypothesis assessment and model audits. a, RMS Piezo1 current difference for the direct signed-force versus WSS and exposure versus absolute-WSS comparisons; the dashed line is the predeclared 3.3 pA detection threshold. b, Number of arteries exceeding the current threshold for each hypothesis–target pair; the dashed line marks the required four arteries. c, Change in RMS current relative to zero external forcing. d, Maximum aggregate current difference between signed-force and exposure pathways; the dashed line is the numerical degeneracy tolerance.\n\nFigure 3 | Parametric robustness of current-based conclusions. a–e, Median six-artery H3 current difference while varying localization area, force-transfer fraction, channel count, apical channel fraction, and fast viscoelastic fraction one parameter at a time around the frozen reference parameterization. The dashed line is the 3.3 pA threshold. f, Maximum fraction of a cardiac cycle at the pressure ceiling as that ceiling is varied.\n"""
    (figure_root / "figure_legends.txt").write_text(legends, encoding="utf-8")
    produced.append(figure_root / "figure_legends.txt")
    return produced


def _checksums(root: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.json":
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            records[str(path.relative_to(root))] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    return records


def run_production(
    *, package_root: Path, output_root: Path, profile: str, repository_commit: str
) -> dict[str, Any]:
    package_root = package_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    study_root = output_root / "scientific_study"
    parameter_root = output_root / "parameter_study"
    figure_root = output_root / "figures"
    provenance_root = output_root / "provenance"; provenance_root.mkdir()
    environment_root = output_root / "environment"; environment_root.mkdir()

    started = time.perf_counter()
    study_manifest = run_scientific_study(
        package_root=package_root,
        output_root=study_root,
        profile=profile,
        calibration_path=package_root / "configs" / "literature_calibration.json",
    )
    if study_manifest.get("status") != "completed_with_claims_disabled":
        raise RuntimeError(json.dumps(study_manifest, indent=2, sort_keys=True))

    model_manifest = json.loads((study_root / "model_outputs" / "manifest.json").read_text())
    summary = pd.read_csv(study_root / "model_outputs" / "six_artery_summary.csv")
    if set(summary["artery_id"].unique()) != EXPECTED_ARTERIES:
        raise RuntimeError("Unexpected artery set")
    if model_manifest.get("profile") != "full":
        raise RuntimeError("The production study did not use the full profile")
    if float(model_manifest["endpoint_reference"]["current_detection_limit_pa"]) != CURRENT_THRESHOLD_PA:
        raise RuntimeError("Current threshold differs from the predeclared value")
    waveform_archive = np.load(study_root / "model_outputs" / "waveforms.npz")
    lengths = {len(waveform_archive[name]) for name in waveform_archive.files if name.endswith("_time_s")}
    if lengths != {2048}:
        raise RuntimeError(f"Unexpected waveform lengths: {sorted(lengths)}")

    parametric_results, parameter_summary = run_parameter_ensemble(
        package_root=package_root, study_root=study_root, output_root=parameter_root, profile=profile
    )
    figure_paths = generate_nature_figures(
        study_root=study_root, parameter_root=parameter_root, figure_root=figure_root
    )
    assessment = json.loads((study_root / "assessment" / "scientific_assessment.json").read_text())
    _write_json(provenance_root / "run_configuration.json", {
        "repository_commit": repository_commit,
        "numerical_profile": profile,
        "current_threshold_pa": CURRENT_THRESHOLD_PA,
        "minimum_passing_arteries": MINIMUM_PASSING_ARTERIES,
        "parameter_scenarios": int(parameter_summary.shape[0]),
        "nature_figure_specification": {
            "width_mm": NATURE_DOUBLE_COLUMN_MM,
            "maximum_height_mm": NATURE_MAX_HEIGHT_MM,
            "body_font_pt": "5-7",
            "panel_label_pt": 8,
            "minimum_line_width_pt": 1,
            "formats": ["pdf", "svg", "png_600dpi", "tiff_600dpi_lzw"],
        },
    })
    _write_json(environment_root / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            dist.metadata.get("Name", "unknown"): dist.version
            for dist in importlib.metadata.distributions() if dist.metadata.get("Name")
        },
    })
    manifest = {
        "workflow": "picoNewton_v4_full_parametric_colab",
        "status": "completed_with_claims_disabled",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": time.perf_counter() - started,
        "package_commit": repository_commit,
        "study_outcome": assessment["study_outcome"],
        "claims_enabled": False,
        "parameter_scenarios": int(parameter_summary.shape[0]),
        "parameter_artery_rows": int(parametric_results.shape[0]),
        "figure_files": sorted(path.name for path in figure_paths),
        "checksummed_files": 0,
    }
    manifest_path = output_root / "FINAL_MANIFEST.json"
    _write_json(manifest_path, manifest)
    records = _checksums(output_root)
    manifest["checksummed_files"] = len(records)
    _write_json(manifest_path, manifest)
    records["FINAL_MANIFEST.json"] = {
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "bytes": manifest_path.stat().st_size,
    }
    _write_json(output_root / "SHA256SUMS.json", records)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("full",), default="full")
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args()
    manifest = run_production(
        package_root=args.package_root,
        output_root=args.output,
        profile=args.profile,
        repository_commit=args.repository_commit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0 if manifest["status"] == "completed_with_claims_disabled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
