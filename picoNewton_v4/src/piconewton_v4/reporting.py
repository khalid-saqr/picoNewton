"""Standard tables and figures for a completed picoNewton_v4 run."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def generate_standard_figures(output_root: Path, figure_root: Path) -> list[Path]:
    """Generate compact, free-tier-safe figures from workflow outputs."""
    output_root = Path(output_root)
    figure_root = Path(figure_root)
    figure_root.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(output_root / "six_artery_summary.csv")
    effects = pd.read_csv(output_root / "hypothesis_effects.csv")
    arrays = np.load(output_root / "waveforms.npz")
    produced: list[Path] = []

    pathway_order = ["wss", "signed", "exposure", "vector"]
    current = summary[summary["pathway"].isin(pathway_order)].copy()
    pivot = current.pivot(index="artery_id", columns="pathway", values="current_rms_pa")
    pivot = pivot.reindex(columns=[c for c in pathway_order if c in pivot.columns])
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("RMS Piezo1 current [pA]")
    ax.set_xlabel("Artery")
    ax.set_title("Six-artery mechanosensory current by pathway")
    ax.figure.tight_layout()
    path = figure_root / "six_artery_current_rms.png"
    ax.figure.savefig(path, dpi=180)
    plt.close(ax.figure)
    produced.append(path)

    h3 = effects[effects["hypothesis"].isin(["H3a", "H3b"])].copy()
    if not h3.empty:
        labels = h3["artery_id"].astype(str) + " | " + h3["target"].astype(str)
        fig, ax = plt.subplots(figsize=(11, max(4, 0.25 * len(h3))))
        ax.barh(labels, h3["current_rms_difference_pa"])
        ax.set_xlabel("RMS current difference [pA]")
        ax.set_title("H3 direct and WSS-surrogate current differences")
        fig.tight_layout()
        path = figure_root / "h3_current_differences.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        produced.append(path)

    artery_ids = sorted({key.split("_time_s")[0] for key in arrays.files if key.endswith("_time_s")})
    for artery_id in artery_ids:
        time = arrays[f"{artery_id}_time_s"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for pathway in pathway_order:
            key = f"{artery_id}_{pathway}_current_pA"
            if key in arrays.files:
                ax.plot(time, arrays[key], label=pathway)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Piezo1 current [pA]")
        ax.set_title(f"{artery_id}: current waveforms")
        ax.legend()
        fig.tight_layout()
        path = figure_root / f"{_safe_name(artery_id)}_current_waveforms.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        produced.append(path)

    return produced


__all__ = ["generate_standard_figures"]
