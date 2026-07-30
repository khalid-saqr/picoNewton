from __future__ import annotations

from pathlib import Path

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


def _save(figure: plt.Figure, root: Path, stem: str, dpi: int) -> list[Path]:
    paths = []
    for extension in ("png", "pdf"):
        path = root / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def create_figures(output_root: str | Path, dpi: int = 300) -> list[Path]:
    output_root = Path(output_root).resolve()
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    atlas = pd.read_csv(output_root / "artery_atlas.csv")
    crossed = pd.read_csv(output_root / "crossed_susceptibility.csv")
    controls = pd.read_csv(output_root / "waveform_controls.csv")
    pairs = pd.read_csv(output_root / "harmonic_pair_attribution.csv")
    predictions = pd.read_csv(output_root / "reduced_law_validation.csv")
    robustness = pd.read_csv(output_root / "constitutive_robustness.csv")

    atlas = atlas.set_index("artery_id").loc[ARTERY_ORDER].reset_index()
    labels = atlas["artery_name"].tolist()

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.loglog(atlas["alpha"], atlas["phi_rms"], "o-")
    for row in atlas.itertuples(index=False):
        axis.annotate(
            row.artery_name,
            (row.alpha, row.phi_rms),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axis.set_xlabel("Womersley number")
    axis.set_ylabel("RMS waveform susceptibility")
    axis.set_title("Six-artery susceptibility atlas")
    axis.grid(True, which="both", alpha=0.3)
    created.extend(_save(figure, figure_root, "figure_1_susceptibility_atlas", dpi))

    physiological = crossed[crossed["condition"] == "physiological"].copy()
    matrix = physiological.pivot(
        index="vessel_id", columns="waveform_id", values="phi_rms"
    )
    matrix = matrix.loc[ARTERY_ORDER, ARTERY_ORDER]
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    image = axis.imshow(matrix.to_numpy(), aspect="auto")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlabel("Transferred pressure waveform")
    axis.set_ylabel("Vessel response")
    axis.set_title("Crossed vessel-waveform susceptibility")
    figure.colorbar(image, ax=axis, label="RMS susceptibility")
    created.extend(_save(figure, figure_root, "figure_2_crossed_matrix", dpi))

    fundamental = controls[controls["control"] == "remove_h1_rms_matched"].copy()
    fundamental = fundamental.set_index("artery_id").loc[ARTERY_ORDER].reset_index()
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(labels, 100.0 * (1.0 - fundamental["relative_to_native"]))
    axis.set_ylabel("Reduction after removing fundamental (%)")
    axis.set_title("RMS-matched fundamental-harmonic removal")
    axis.tick_params(axis="x", rotation=45)
    created.extend(_save(figure, figure_root, "figure_3_waveform_controls", dpi))

    dominant = pairs[
        (pairs["output_frequency"].isin([0, 1, 2])) & (pairs["rank"] == 1)
    ]
    dominant = dominant.sort_values(["artery_id", "output_frequency"])
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(dominant))
    axis.bar(x, dominant["fraction_of_absolute_pair_sum"])
    axis.set_xticks(
        x,
        [
            f"{row.artery_id}\nq={row.output_frequency}: ({row.m},{row.n})"
            for row in dominant.itertuples(index=False)
        ],
        rotation=65,
        ha="right",
    )
    axis.set_ylabel("Dominant-pair share")
    axis.set_title("Dominant harmonic pairs for principal output frequencies")
    created.extend(_save(figure, figure_root, "figure_4_harmonic_pairs", dpi))

    figure, axis = plt.subplots(figsize=(6.5, 5.2))
    axis.scatter(
        predictions["exact_phi_rms"],
        predictions["predicted_phi_rms"],
        s=10,
        alpha=0.45,
    )
    low = min(
        predictions["exact_phi_rms"].min(),
        predictions["predicted_phi_rms"].min(),
    )
    high = max(
        predictions["exact_phi_rms"].max(),
        predictions["predicted_phi_rms"].max(),
    )
    axis.plot([low, high], [low, high], "--", linewidth=1)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Full operator susceptibility")
    axis.set_ylabel("Rank-one prediction")
    axis.set_title("Leave-one-artery-out reduced-law validation")
    axis.grid(True, which="both", alpha=0.3)
    created.extend(_save(figure, figure_root, "figure_5_reduced_law", dpi))

    non_null = robustness[~robustness["null_control"]].copy()
    summary = non_null.groupby("constitutive_path").agg(
        median_shape_error=("normalised_shape_relative_l2", "median"),
        maximum_amplitude_ratio=("relative_amplitude_to_reciprocal", "max"),
    )
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(summary))
    axis.bar(
        x - 0.2,
        100.0 * summary["median_shape_error"],
        width=0.4,
        label="Median shape error (%)",
    )
    axis.bar(
        x + 0.2,
        100.0 * np.abs(summary["maximum_amplitude_ratio"] - 1.0),
        width=0.4,
        label="Maximum amplitude departure (%)",
    )
    axis.set_xticks(x, summary.index, rotation=45, ha="right")
    axis.set_title("Constitutive robustness: shape and amplitude")
    axis.legend()
    created.extend(_save(figure, figure_root, "figure_6_constitutive_robustness", dpi))

    return created
