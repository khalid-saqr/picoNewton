# ruff: noqa: E501
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARTERY_ORDER = [
    "aortic_root",
    "thoracic_aorta",
    "femoral",
    "carotid",
    "iliac",
    "brachial",
]


def _read(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.is_file():
        raise RuntimeError(f"publication source is missing: {path}")
    return pd.read_csv(path)


def _save(fig: plt.Figure, destination: Path, formats: tuple[str, ...], dpi: int) -> list[Path]:
    outputs = []
    for extension in formats:
        path = destination.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def _labels(frame: pd.DataFrame) -> list[str]:
    column = "artery_name" if "artery_name" in frame.columns else "artery_id"
    return frame[column].astype(str).tolist()


def figure1(step4: Path, destination: Path, source_root: Path, formats, dpi) -> tuple[list[Path], list[Path]]:
    continuity = _read(step4, "step3_waveform_continuity.csv")
    slopes = _read(step4, "order_slopes.csv")
    validity = _read(step4, "validity_domains.csv")
    continuity.to_csv(source_root / "figure01_parent_continuity.csv", index=False)
    slopes.to_csv(source_root / "figure01_order_slopes.csv", index=False)
    validity.to_csv(source_root / "figure01_validity_domains.csv", index=False)
    labels = _labels(slopes)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    continuity_columns = [
        "isotropic_waveform_relative_l2",
        "epsilon_0p1_waveform_relative_l2",
    ]
    for column in continuity_columns:
        axes[0].plot(
            x,
            continuity[column],
            marker="o",
            label=column.replace("_waveform_relative_l2", "").replace("epsilon_0p1", r"$\varepsilon=0.1$"),
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, _labels(continuity), rotation=35, ha="right")
    axes[0].set_ylabel("Relative waveform error")
    axes[0].set_title("Step 3–4 continuity")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(x, slopes["ut_order"], marker="o", label=r"$U_\theta$")
    axes[1].plot(x, slopes["uz_correction_order"], marker="s", label=r"$U_z-U_z^{(0)}$")
    axes[1].plot(x, slopes["signed_force_excess_order"], marker="^", label=r"$\Delta F_s$")
    axes[1].axhline(1.0, linewidth=0.8, linestyle="--")
    axes[1].axhline(2.0, linewidth=0.8, linestyle="--")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylabel("Measured asymptotic order")
    axes[1].set_title("Perturbative hierarchy")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].bar(x - 0.18, validity["force_valid_epsilon_max_1pct"], width=0.36, label="Force")
    axes[2].bar(x + 0.18, validity[["ut_valid_epsilon_max_1pct", "uz_valid_epsilon_max_1pct"]].min(axis=1), width=0.36, label="Fields")
    axes[2].set_xticks(x, _labels(validity), rotation=35, ha="right")
    axes[2].set_ylabel(r"Maximum validated $\varepsilon$")
    axes[2].set_ylim(0, 0.11)
    axes[2].set_title("One-percent validity domain")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Parent continuity and weak-anisotropy hierarchy")
    return _save(fig, destination, formats, dpi), [
        source_root / "figure01_parent_continuity.csv",
        source_root / "figure01_order_slopes.csv",
        source_root / "figure01_validity_domains.csv",
    ]


def figure2(step5: Path, destination: Path, source_root: Path, formats, dpi) -> tuple[list[Path], list[Path]]:
    closure = _read(step5, "kernel_closure.csv")
    dominant = _read(step5, "dominant_pairs.csv")
    closure.to_csv(source_root / "figure02_kernel_closure.csv", index=False)
    selected = dominant.query("kernel_type == 'second_order' and rank == 1 and q in [0, 1, 2]").copy()
    selected.to_csv(source_root / "figure02_dominant_pairs.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    error_columns = [c for c in closure.columns if c.endswith("relative_l2") or c.endswith("relative_error")]
    if error_columns:
        maxima = closure.groupby("kernel_type")[error_columns].max().max(axis=1)
        axes[0].bar(np.arange(len(maxima)), maxima.values)
        axes[0].set_xticks(np.arange(len(maxima)), [str(v).replace("second_order", "2nd").replace("exact_excess", "exact") for v in maxima.index], rotation=30, ha="right")
        axes[0].set_yscale("log")
    axes[0].set_ylabel("Maximum relative closure error")
    axes[0].set_title("Direct–kernel equivalence")
    if not selected.empty:
        pivot = selected.pivot_table(index="artery_name", columns="q", values="fraction_of_pairwise_absolute_sum", aggfunc="max")
        image = axes[1].imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="Greys")
        axes[1].set_yticks(range(len(pivot.index)), pivot.index)
        axes[1].set_xticks(range(len(pivot.columns)), [f"q={q}" for q in pivot.columns])
        fig.colorbar(image, ax=axes[1], label="Dominant-pair absolute share")
    axes[1].set_title("Dominant harmonic pairs")
    fig.suptitle("Exact harmonic interaction law")
    return _save(fig, destination, formats, dpi), [source_root / "figure02_kernel_closure.csv", source_root / "figure02_dominant_pairs.csv"]


def figure3(step7: Path, destination: Path, source_root: Path, formats, dpi) -> tuple[list[Path], list[Path]]:
    controls = _read(step7, "native_waveform_controls.csv")
    families = _read(step7, "causal_waveform_families.csv")
    controls.to_csv(source_root / "figure03_native_controls.csv", index=False)
    families.to_csv(source_root / "figure03_causal_families.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    if {"family", "relative_to_native_rms"}.issubset(controls.columns):
        summary = controls.groupby("family")["relative_to_native_rms"].agg(["min", "median", "max"])
    else:
        value = "phi_rms_ratio_to_native" if "phi_rms_ratio_to_native" in controls.columns else controls.select_dtypes("number").columns[-1]
        summary = controls.groupby("family")[value].agg(["min", "median", "max"])
    x = np.arange(len(summary))
    axes[0].errorbar(x, summary["median"], yerr=[summary["median"] - summary["min"], summary["max"] - summary["median"]], fmt="o", capsize=3)
    axes[0].axhline(1.0, linestyle="--", linewidth=0.8)
    axes[0].set_xticks(x, summary.index.astype(str), rotation=35, ha="right")
    axes[0].set_ylabel("Susceptibility relative to native")
    axes[0].set_title("Amplitude, sign and phase controls")
    value_col = "phi_rms" if "phi_rms" in families.columns else families.select_dtypes("number").columns[-1]
    family_summary = families.groupby("family")[value_col].median().sort_values(ascending=False)
    axes[1].bar(np.arange(len(family_summary)), family_summary.values)
    axes[1].set_xticks(np.arange(len(family_summary)), family_summary.index.astype(str), rotation=35, ha="right")
    axes[1].set_ylabel("Median susceptibility")
    axes[1].set_title("Controlled waveform families")
    fig.suptitle("Waveform organisation controls")
    return _save(fig, destination, formats, dpi), [source_root / "figure03_native_controls.csv", source_root / "figure03_causal_families.csv"]


def figure4(step4: Path, step6: Path, destination: Path, source_root: Path, formats, dpi) -> tuple[list[Path], list[Path]]:
    native = _read(step6, "native_susceptibility.csv")
    sweep = _read(step4, "epsilon_sweep.csv")
    native.to_csv(source_root / "figure04_native_susceptibility.csv", index=False)
    sweep.to_csv(source_root / "figure04_epsilon_sweep.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].scatter(native["alpha"], native["phi_rms"], s=55)
    for row in native.itertuples(index=False):
        axes[0].annotate(row.artery_id, (row.alpha, row.phi_rms), fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Womersley number $\alpha$")
    axes[0].set_ylabel(r"RMS susceptibility $\Phi_{2,\mathrm{rms}}$")
    axes[0].set_title("Native susceptibility landscape")
    grouped = sweep.groupby("epsilon")[["signed_excess_waveform_relative_l2", "signed_excess_rms_relative_error", "signed_excess_peak_relative_error"]].max()
    for column in grouped.columns:
        axes[1].plot(grouped.index, grouped[column], marker="o", label=column.replace("signed_excess_", "").replace("_relative", ""))
    axes[1].axhline(0.01, linestyle="--", linewidth=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\varepsilon$")
    axes[1].set_ylabel("Maximum relative error")
    axes[1].set_title("Perturbative validity")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Susceptibility landscape and validity")
    return _save(fig, destination, formats, dpi), [source_root / "figure04_native_susceptibility.csv", source_root / "figure04_epsilon_sweep.csv"]


def figure5(step5: Path, step6: Path, destination: Path, source_root: Path, formats, dpi) -> tuple[list[Path], list[Path]]:
    native = _read(step6, "native_susceptibility.csv")
    dominant = _read(step5, "dominant_pairs.csv")
    native.to_csv(source_root / "figure05_susceptibility_atlas.csv", index=False)
    dominant.query("kernel_type == 'second_order' and rank == 1 and q in [0, 1, 2]").to_csv(source_root / "figure05_atlas_dominant_pairs.csv", index=False)
    order = native.sort_values("phi_rms", ascending=False)
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(x - 0.18, order["phi_rms"], width=0.36, label="RMS")
    axes[0].bar(x + 0.18, order["phi_peak_abs"], width=0.36, label="Peak")
    axes[0].set_xticks(x, order["artery_name"], rotation=35, ha="right")
    axes[0].set_ylabel("Dimensionless susceptibility")
    axes[0].set_title("Six-artery susceptibility")
    axes[0].legend(frameon=False)
    axes[1].bar(x - 0.18, order["outward_duty"], width=0.36, label="Outward")
    axes[1].bar(x + 0.18, order["inward_duty"], width=0.36, label="Inward")
    axes[1].set_xticks(x, order["artery_name"], rotation=35, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Cycle fraction")
    axes[1].set_title("Directional duty fractions")
    axes[1].legend(frameon=False)
    fig.suptitle("Six-artery physiological susceptibility atlas")
    return _save(fig, destination, formats, dpi), [source_root / "figure05_susceptibility_atlas.csv", source_root / "figure05_atlas_dominant_pairs.csv"]


def figure6(step6: Path, step7: Path, destination: Path, source_root: Path, formats, dpi) -> tuple[list[Path], list[Path]]:
    crossed = _read(step7, "crossed_susceptibility.csv")
    critical = _read(step6, "critical_anisotropy.csv")
    crossed.to_csv(source_root / "figure06_crossed_matrix.csv", index=False)
    critical.to_csv(source_root / "figure06_critical_anisotropy.csv", index=False)
    phys = crossed.query("matrix_type == 'physiological'")
    matrix = phys.pivot(index="vessel_name", columns="waveform_name", values="phi_rms")
    rms_1pn = critical.query("primary_metric == True and benchmark_pn == 1.0").copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    image = axes[0].imshow(matrix.to_numpy(), aspect="auto", cmap="Greys")
    axes[0].set_yticks(range(len(matrix.index)), matrix.index)
    axes[0].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axes[0].set_title("Physiological crossed susceptibility")
    fig.colorbar(image, ax=axes[0], label=r"$\Phi_{2,\mathrm{rms}}$")
    x = np.arange(len(rms_1pn))
    axes[1].bar(x, rms_1pn["perturbative_epsilon_critical"])
    axes[1].scatter(x, rms_1pn["validated_domain_max"], marker="_", s=250, label="Validated maximum")
    axes[1].axhline(1.0, linestyle="--", linewidth=0.8, label="Constitutive limit")
    axes[1].set_xticks(x, rms_1pn["artery_name"], rotation=35, ha="right")
    axes[1].set_ylabel(r"Critical anisotropy $\varepsilon$")
    axes[1].set_title("Formal 1 pN critical anisotropy")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Crossed matrix and critical-anisotropy prediction")
    return _save(fig, destination, formats, dpi), [source_root / "figure06_crossed_matrix.csv", source_root / "figure06_critical_anisotropy.csv"]


def build_supplementary_figures(step_roots: dict[int, Path], output_root: Path, formats: tuple[str, ...], dpi: int) -> dict[str, list[Path]]:
    figure_root = output_root / "figures" / "supplementary"
    source_root = output_root / "figures" / "source"
    figure_root.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, list[Path]] = {"figures": [], "sources": []}

    selection = _read(step_roots[8], "model_selection.csv")
    family = _read(step_roots[8], "compact_law_family_summary.csv")
    selection.to_csv(source_root / "figure_s01_model_selection.csv", index=False)
    family.to_csv(source_root / "figure_s01_family_validation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    modal = selection.query("rank >= 1").sort_values("rank")
    axes[0].plot(modal["rank"], modal["median_relative_error"], marker="o", label="Median")
    axes[0].plot(modal["rank"], modal["maximum_relative_error"], marker="s", label="Maximum")
    axes[0].set_xticks(modal["rank"])
    axes[0].set_xlabel("Universal-kernel rank")
    axes[0].set_ylabel("Held-out relative error")
    axes[0].set_title("Rank selection")
    axes[0].legend(frameon=False)
    family_col = "family" if "family" in family.columns else family.columns[1]
    axes[1].bar(np.arange(len(family)), family["median_relative_error"])
    axes[1].set_xticks(np.arange(len(family)), family[family_col].astype(str), rotation=35, ha="right")
    axes[1].set_ylabel("Median held-out error")
    axes[1].set_title("Waveform-family validation")
    fig.suptitle("Reduced-law model selection")
    outputs["figures"].extend(_save(fig, figure_root / "figure_s01_reduced_law_validation", formats, dpi))
    outputs["sources"].extend([source_root / "figure_s01_model_selection.csv", source_root / "figure_s01_family_validation.csv"])

    metrics = _read(step_roots[9], "constitutive_path_metrics.csv")
    metrics.to_csv(source_root / "figure_s02_constitutive_robustness.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x = np.arange(len(metrics))
    ax.bar(x - 0.2, metrics["shape_maximum"], width=0.4, label="Shape")
    ax.bar(x + 0.2, metrics["frozen_amplitude_maximum"], width=0.4, label="Frozen amplitude")
    ax.axhline(0.20, linestyle="--", linewidth=0.8, label="20% shape gate")
    ax.set_xticks(x, metrics["path"], rotation=35, ha="right")
    ax.set_ylabel("Maximum relative error")
    ax.set_yscale("log")
    ax.set_title("Constitutive robustness: interaction shape versus amplitude")
    ax.legend(frameon=False)
    outputs["figures"].extend(_save(fig, figure_root / "figure_s02_constitutive_robustness", formats, dpi))
    outputs["sources"].append(source_root / "figure_s02_constitutive_robustness.csv")

    eta = _read(step_roots[9], "eta_robustness.csv")
    resolution = _read(step_roots[9], "resolution_robustness.csv")
    eta.to_csv(source_root / "figure_s03_eta_robustness.csv", index=False)
    resolution.to_csv(source_root / "figure_s03_resolution_robustness.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    eta_summary = eta.groupby("eta_multiplier")["relative_error"].agg(["median", "max"])
    axes[0].plot(eta_summary.index, eta_summary["median"], marker="o", label="Median")
    axes[0].plot(eta_summary.index, eta_summary["max"], marker="s", label="Maximum")
    axes[0].set_xlabel(r"Near-wall thickness multiplier")
    axes[0].set_ylabel("Frozen-law relative error")
    axes[0].set_title("Control-volume sensitivity")
    axes[0].legend(frameon=False)
    res_summary = resolution.groupby(["radial_order", "quadrature_nodes"])["relative_change"].max().reset_index()
    labels = [f"N={r.radial_order}, Q={r.quadrature_nodes}" for r in res_summary.itertuples(index=False)]
    axes[1].bar(np.arange(len(res_summary)), res_summary["relative_change"])
    axes[1].set_xticks(np.arange(len(res_summary)), labels, rotation=25, ha="right")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Maximum relative change")
    axes[1].set_title("Independent resolution checks")
    fig.suptitle("Near-wall and numerical robustness")
    outputs["figures"].extend(_save(fig, figure_root / "figure_s03_numerical_robustness", formats, dpi))
    outputs["sources"].extend([source_root / "figure_s03_eta_robustness.csv", source_root / "figure_s03_resolution_robustness.csv"])
    return outputs


def build_main_figures(step_roots: dict[int, Path], output_root: Path, formats: tuple[str, ...], dpi: int) -> dict[str, list[Path]]:
    figure_root = output_root / "figures" / "main"
    source_root = output_root / "figures" / "source"
    figure_root.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    builders: list[tuple[str, Any]] = [
        ("figure_01_parent_continuity_hierarchy", lambda p: figure1(step_roots[4], p, source_root, formats, dpi)),
        ("figure_02_exact_interaction_kernel", lambda p: figure2(step_roots[5], p, source_root, formats, dpi)),
        ("figure_03_waveform_controls", lambda p: figure3(step_roots[7], p, source_root, formats, dpi)),
        ("figure_04_susceptibility_validity", lambda p: figure4(step_roots[4], step_roots[6], p, source_root, formats, dpi)),
        ("figure_05_six_artery_atlas", lambda p: figure5(step_roots[5], step_roots[6], p, source_root, formats, dpi)),
        ("figure_06_crossed_matrix_thresholds", lambda p: figure6(step_roots[6], step_roots[7], p, source_root, formats, dpi)),
    ]
    result: dict[str, list[Path]] = {"figures": [], "sources": []}
    for name, builder in builders:
        figures, sources = builder(figure_root / name)
        result["figures"].extend(figures)
        result["sources"].extend(sources)
    return result
