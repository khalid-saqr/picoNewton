"""Publication figures and archive-ready dataset export."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr

from .model import (
    ArteryCase, EndothelialControlVolume, FluidProperties, HydrodynamicConfig,
    SensorConfig, V2_ARTERY_CASES, compute_hydrodynamics, lamb_work,
    periodic_sensor_solution, rms_difference, signal_metrics, wss_work,
)
from .study_io import StudyStore
from .workflow_common import (
    DEFAULT_SEED, NUMERICAL_SENSOR_UNCERTAINTY, _phase_distance, _rms,
    _sensor_from_work, run_hydrodynamic_cases,
)

def generate_publication_figures(
    output_dir: str | Path,
    waveform_table: pd.DataFrame,
    control_table: pd.DataFrame,
    parameter_grid: pd.DataFrame,
    surrogate_table: pd.DataFrame,
    gate_table: pd.DataFrame,
    dominance_table: pd.DataFrame,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Figure 1: signed force across all arteries.
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for artery_name, group in waveform_table.groupby("artery_name"):
        ax.plot(group["time_cycle"], group["force_signed_n"] * 1e12, label=artery_name)
    ax.set_xlabel("Cardiac cycle, t/T")
    ax.set_ylabel("Signed Lamb force, pN")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = output / "Figure1_force_waveforms.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    created.extend([path, path.with_suffix(".png")])

    # Figure 2: mechanosensory channels for a representative artery.
    representative = "carotid" if "carotid" in set(waveform_table.artery_id) else waveform_table.artery_id.iloc[0]
    group = waveform_table[waveform_table["artery_id"] == representative]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(group["time_cycle"], group["p_WSS"], label="WSS only")
    ax.plot(group["time_cycle"], group["p_Lamb"], label="Lamb only")
    ax.plot(group["time_cycle"], group["p_parallel"], label="Parallel")
    ax.set_xlabel("Cardiac cycle, t/T")
    ax.set_ylabel("Active-state probability")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output / "Figure2_sensor_channels.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    created.extend([path, path.with_suffix(".png")])

    # Figure 3: median observability map.
    pivot = parameter_grid.groupby(
        ["coupling_length_nm", "relaxation_time_s"]
    )["effect_parallel_vs_WSS"].median().unstack()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    image = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            np.log10(pivot.columns.min()),
            np.log10(pivot.columns.max()),
            np.log10(pivot.index.min()),
            np.log10(pivot.index.max()),
        ],
    )
    ax.set_xlabel(r"$\log_{10}(\tau_0/\mathrm{s})$")
    ax.set_ylabel(r"$\log_{10}(d_L/\mathrm{nm})$")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Median RMS response difference")
    fig.tight_layout()
    path = output / "Figure3_observability_map.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    created.extend([path, path.with_suffix(".png")])

    # Figure 4: nominal controls.
    control_summary = control_table.groupby("control_id")["rms"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.barh(control_summary.index, control_summary.values)
    ax.set_xlabel("Median sensor RMS")
    fig.tight_layout()
    path = output / "Figure4_control_matrix.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    created.extend([path, path.with_suffix(".png")])

    # Figure 5: WSS surrogate.
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(surrogate_table["artery_name"], surrogate_table["rms_residual"])
    ax.axhline(0.005, linestyle="--", linewidth=1.2, label="E2 threshold")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Best-fit WSS residual, RMS")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output / "Figure5_wss_surrogate.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    created.extend([path, path.with_suffix(".png")])

    # Figure 6: effect gate matrix.
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    values = gate_table["passed"].astype(int).to_numpy().reshape(1, -1)
    ax.imshow(values, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(gate_table)), gate_table["criterion_id"])
    ax.set_yticks([0], ["Run"])
    for column, passed in enumerate(values[0]):
        ax.text(column, 0, "PASS" if passed else "FAIL", ha="center", va="center")
    fig.tight_layout()
    path = output / "Figure6_effect_gates.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    created.extend([path, path.with_suffix(".png")])

    # Figure 7: high-harmonic and anisotropy effects.
    summary = parameter_grid.groupby("artery_name")[
        ["high_harmonic_abs", "anisotropy_effect"]
    ].median()
    x = np.arange(len(summary))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - width / 2, summary["high_harmonic_abs"], width, label="High-harmonic")
    ax.bar(x + width / 2, summary["anisotropy_effect"], width, label="Anisotropy")
    ax.set_xticks(x, summary.index, rotation=30, ha="right")
    ax.set_ylabel("Median absolute RMS effect")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output / "Figure7_specificity_effects.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    created.extend([path, path.with_suffix(".png")])

    # Figure 8: parameter dominance.
    dominance = dominance_table.pivot(
        index="artery_name", columns="parameter", values="spearman_rho"
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    image = ax.imshow(dominance.values, aspect="auto", vmin=-1, vmax=1)
    ax.set_yticks(range(len(dominance)), dominance.index)
    ax.set_xticks(range(len(dominance.columns)), dominance.columns, rotation=30, ha="right")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Spearman correlation")
    fig.tight_layout()
    path = output / "Figure8_parameter_dominance.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    created.extend([path, path.with_suffix(".png")])
    return created


def export_publication_dataset(
    store: StudyStore,
    run_id: str,
    *,
    waveform_table: pd.DataFrame,
    control_table: pd.DataFrame,
    parameter_grid: pd.DataFrame,
    surrogate_table: pd.DataFrame,
    gate_table: pd.DataFrame,
    dominance_table: pd.DataFrame,
    hydrodynamics: dict[str, dict[str, Any]],
) -> list[Path]:
    paths: list[Path] = []
    tables = {
        "summaries/nominal_waveforms.csv": waveform_table,
        "summaries/control_results.csv": control_table,
        "summaries/parameter_grid.csv": parameter_grid,
        "summaries/wss_surrogate.csv": surrogate_table,
        "summaries/effect_gates.csv": gate_table,
        "summaries/parameter_dominance.csv": dominance_table,
    }
    for relative, table in tables.items():
        path = store.write_csv(f"runs/{run_id}/{relative}", table)
        store.register_file(run_id, relative, "output")
        paths.append(path)

    groups: dict[str, dict[str, np.ndarray]] = {}
    for artery_id, hydro in hydrodynamics.items():
        groups[artery_id] = {
            "time_cycle": np.asarray(hydro["time_cycle"]),
            "time_s": np.asarray(hydro["time_s"]),
            "force_signed_n": np.asarray(hydro["force_signed_n"]),
            "force_exposure_n": np.asarray(hydro["force_exposure_n"]),
            "wall_shear_pa": np.asarray(hydro["wall_shear_pa"]),
        }
    hdf5_path = store.write_hdf5(f"runs/{run_id}/fields/hydrodynamic_signals.h5", groups)
    store.register_file(run_id, "fields/hydrodynamic_signals.h5", "output")
    paths.append(hdf5_path)
    checksum_path = store.write_checksums(
        f"runs/{run_id}", "provenance/checksums.sha256"
    )
    paths.append(checksum_path)
    return paths
