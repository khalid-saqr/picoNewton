from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D

ARTERY_ORDER = [
    "carotid",
    "brachial",
    "aortic_root",
    "thoracic_aorta",
    "iliac",
    "femoral",
]
WAVEFORM_LABELS = {
    "carotid": "CA",
    "brachial": "BA",
    "aortic_root": "AoR",
    "thoracic_aorta": "TA",
    "iliac": "IA",
    "femoral": "FA",
}
WIDTH_MM = 180.0
MAX_HEIGHT_MM = 112.0
FONT_PT = 7.0
PANEL_PT = 8.0
LINE_PT = 1.0
SEQ_CMAP = "cividis"
DIV_CMAP = "RdBu_r"


def _inch(mm: float) -> float:
    return mm / 25.4


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FONT_PT,
            "axes.labelsize": FONT_PT,
            "axes.titlesize": FONT_PT,
            "xtick.labelsize": FONT_PT,
            "ytick.labelsize": FONT_PT,
            "legend.fontsize": FONT_PT,
            "axes.linewidth": 0.8,
            "lines.linewidth": LINE_PT,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _label(axes: Iterable[plt.Axes]) -> None:
    for letter, axis in zip("abcdefghijklmnopqrstuvwxyz", axes, strict=False):
        axis.text(
            -0.13,
            1.04,
            letter,
            transform=axis.transAxes,
            fontsize=PANEL_PT,
            fontweight="bold",
            va="bottom",
        )


def _clean(axis: plt.Axes, grid: bool = False) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(True, linewidth=0.45, alpha=0.25)
        axis.set_axisbelow(True)


def _save(
    fig: plt.Figure,
    root: Path,
    stem: str,
    dpi: int,
    height: float,
    manuscript_figure: str,
):
    paths = []
    for suffix in ("pdf", "svg", "png"):
        path = root / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=dpi if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.03,
        )
        paths.append(path)
    plt.close(fig)
    return paths, {
        "figure": manuscript_figure,
        "file_stem": stem,
        "width_mm": WIDTH_MM,
        "height_mm": height,
        "font_size_pt": FONT_PT,
        "minimum_line_width_pt": LINE_PT,
        "raster_dpi": dpi,
        "formats": "pdf,svg,png",
    }


def _ordered(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    return frame.set_index(key).loc[ARTERY_ORDER].reset_index()


def _names(frame: pd.DataFrame, key: str, name: str) -> list[str]:
    return frame.drop_duplicates(key).set_index(key).loc[ARTERY_ORDER, name].tolist()


def _scientific_label(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def _native(atlas: pd.DataFrame, root: Path, dpi: int):
    height = 66.0
    atlas = _ordered(atlas, "artery_id")
    names = (
        atlas["artery_name"]
        .str.replace("Aortic Root", "Aortic root")
        .str.replace("Thoracic Aorta", "Thoracic aorta")
    )
    y = np.arange(len(atlas))

    intrinsic_rank = atlas["phi_rms"].rank(method="min", ascending=False)
    force_rank = atlas["predicted_rms_at_epsilon_0p08_pn"].rank(method="min", ascending=False)
    rank_change = intrinsic_rank - force_rank

    fig, axes = plt.subplots(
        1, 2, figsize=(_inch(WIDTH_MM), _inch(height)), constrained_layout=True
    )
    ax = axes[0]
    ax.scatter(atlas["phi_rms"], y, s=29, color="black", zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set(
        xlabel=r"Intrinsic susceptibility, $\Phi_{\mathrm{rms}}$",
        ylabel="Arterial site",
        title="Intrinsic susceptibility",
    )
    for x_value, y_value in zip(atlas["phi_rms"], y, strict=True):
        ax.annotate(
            _scientific_label(x_value),
            (x_value, y_value),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            color="0.25",
        )
    ax.set_xlim(atlas["phi_rms"].min() / 1.7, atlas["phi_rms"].max() * 2.4)
    _clean(ax, True)

    ax = axes[1]
    positive = rank_change > 0
    bars = ax.barh(
        y,
        rank_change,
        color=np.where(positive, "black", "white"),
        edgecolor="black",
        linewidth=1.0,
    )
    for bar, is_positive in zip(bars, positive, strict=True):
        if not is_positive:
            bar.set_hatch("//")
    ax.axvline(0, color="0.2", linewidth=0.8)
    ax.set_yticks(y, [])
    ax.invert_yaxis()
    ax.set(
        xlabel=(
            r"Rank change, $\Delta r=r_\Phi-r_F$"
            "\n"
            r"$\leftarrow$ moves away from rank 1      moves toward rank 1 $\rightarrow$"
        ),
        title="Rank change after dimensional reconstruction",
        xlim=(-6.2, 6.2),
    )
    for index, (change, force) in enumerate(
        zip(rank_change, atlas["predicted_rms_at_epsilon_0p08_pn"], strict=True)
    ):
        if change > 0:
            ax.text(
                change - 0.15,
                index,
                f"+{int(change)}",
                ha="right",
                va="center",
                color="white",
                fontweight="bold",
            )
            ax.text(change + 0.15, index, f"{force:.3f} pN", ha="left", va="center", color="0.25")
        else:
            ax.text(
                change + 0.15, index, f"{int(change)}", ha="left", va="center", fontweight="bold"
            )
            ax.text(change - 0.15, index, f"{force:.3f} pN", ha="right", va="center", color="0.25")
    _clean(ax, True)
    _label(axes)
    return _save(fig, root, "Figure1", dpi, height, "Figure 1")


def _crossed(crossed: pd.DataFrame, root: Path, dpi: int):
    height = 52.0
    selected = crossed[crossed["condition"] == "physiological"]
    matrix = (
        selected.pivot(index="vessel_id", columns="waveform_id", values="phi_rms")
        .loc[ARTERY_ORDER, ARTERY_ORDER]
        .to_numpy()
    )
    row_geometric_mean = np.exp(np.mean(np.log(matrix), axis=1, keepdims=True))
    redistribution = 100.0 * (matrix / row_geometric_mean - 1.0)
    logs = np.log(matrix)
    grand_log_mean = float(np.mean(logs))
    artery_factor = np.exp(np.mean(logs, axis=1, keepdims=True))
    waveform_factor = np.exp(np.mean(logs, axis=0, keepdims=True) - grand_log_mean)
    residual = 100.0 * (matrix / (artery_factor * waveform_factor) - 1.0)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(_inch(WIDTH_MM), _inch(height)),
        gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]},
        constrained_layout=True,
    )
    names = _names(selected, "vessel_id", "vessel_name")
    names = [
        name.replace("Aortic Root", "Aortic root").replace("Thoracic Aorta", "Thoracic aorta")
        for name in names
    ]
    abbreviations = [WAVEFORM_LABELS[item] for item in ARTERY_ORDER]
    diagonal = np.arange(6)

    ax = axes[0]
    image = ax.imshow(
        matrix, cmap=SEQ_CMAP, norm=LogNorm(matrix.min(), matrix.max()), aspect="auto"
    )
    ax.scatter(diagonal, diagonal, s=24, facecolors="white", edgecolors="black", linewidth=0.8)
    ax.set_xticks(diagonal, abbreviations)
    ax.set_yticks(diagonal, names)
    ax.set(xlabel="Waveform source", ylabel="Vessel model", title="Crossed susceptibility")
    fig.colorbar(image, ax=ax, pad=0.025, label=r"$\Phi_{\mathrm{rms}}$")

    ax = axes[1]
    limit = float(np.max(np.abs(redistribution)))
    image = ax.imshow(
        redistribution,
        cmap=DIV_CMAP,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    ax.scatter(diagonal, diagonal, s=18, color="black")
    ax.set_xticks(diagonal, abbreviations)
    ax.set_yticks(diagonal, [])
    ax.set(xlabel="Waveform source", title="Waveform redistribution")
    for row in range(6):
        for column in range(6):
            value = redistribution[row, column]
            colour = "white" if abs(value) > 0.55 * limit else "black"
            ax.text(
                column,
                row,
                f"{value:+.1f}" if value > 0 else f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=5.6,
                color=colour,
            )
    fig.colorbar(image, ax=ax, pad=0.025, label="Departure from row\ngeometric mean (%)")

    ax = axes[2]
    residual_limit = float(np.max(np.abs(residual)))
    image = ax.imshow(
        residual,
        cmap=DIV_CMAP,
        norm=TwoSlopeNorm(vmin=-residual_limit, vcenter=0.0, vmax=residual_limit),
        aspect="auto",
    )
    ax.scatter(diagonal, diagonal, s=18, color="black")
    ax.set_xticks(diagonal, abbreviations)
    ax.set_yticks(diagonal, [])
    ax.set(xlabel="Waveform source", title="Residual from multiplicative separation")
    ax.text(
        0.03,
        0.93,
        rf"max $|$residual$|$ = {residual_limit:.2f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )
    fig.colorbar(image, ax=ax, pad=0.025, label="Residual (%)")
    _label(axes)
    paths, record = _save(fig, root, "Figure2", dpi, height, "Figure 2")
    record.update(
        colormaps=f"{SEQ_CMAP},{DIV_CMAP}", maximum_separation_residual_percent=residual_limit
    )
    return paths, record


def _pair_statistics(
    pairs: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, int, float]]]]:
    table = pairs.copy()
    table["contribution"] = table["contribution_real_n"] + 1j * table["contribution_imag_n"]
    totals = table.groupby(["artery_id", "output_frequency"])["contribution"].transform("sum")
    table["signed_projection"] = (
        100.0
        * np.real(table["contribution"] * np.conj(totals))
        / np.maximum(np.abs(totals) ** 2, np.finfo(float).tiny)
    )

    spectrum = np.empty((6, 13), dtype=float)
    dominant = np.empty((6, 13), dtype=float)
    for row, artery in enumerate(ARTERY_ORDER):
        artery_table = table[table["artery_id"] == artery]
        q0 = float(
            artery_table.loc[artery_table["output_frequency"] == 0, "output_absolute_n"].iloc[0]
        )
        for q in range(13):
            selected = artery_table[artery_table["output_frequency"] == q]
            spectrum[row, q] = float(selected["output_absolute_n"].iloc[0]) / q0
            dominant[row, q] = 100.0 * float(selected["fraction_of_absolute_pair_sum"].max())

    leading: list[list[tuple[int, int, float]]] = []
    for q in range(13):
        selected = table[table["output_frequency"] == q]
        grouped = (
            selected.groupby(["m", "n"], as_index=False)
            .agg(
                mean_absolute_share=("fraction_of_absolute_pair_sum", "mean"),
                mean_signed_projection=("signed_projection", "mean"),
            )
            .sort_values("mean_absolute_share", ascending=False)
        )
        leading.append(
            [
                (int(row.m), int(row.n), float(row.mean_signed_projection))
                for row in grouped.head(3).itertuples()
            ]
        )
    return spectrum, dominant, leading


def _pairs(pairs: pd.DataFrame, root: Path, dpi: int):
    height = 110.0
    spectrum, dominant, leading = _pair_statistics(pairs)
    signed = np.full((13, 6), np.nan)
    pair_labels: list[list[str]] = [["" for _ in range(6)] for _ in range(13)]
    for q, rows in enumerate(leading):
        for rank, (m, n, projection) in enumerate(rows):
            signed[q, rank] = projection
            pair_labels[q][rank] = f"({m},{n})\n{projection:+.0f}"

    fig = plt.figure(figsize=(_inch(WIDTH_MM), _inch(height)), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(0.78, 1.42), height_ratios=(1.0, 0.88))
    axes = [fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])]

    ax = axes[0]
    masked = np.ma.masked_invalid(signed)
    limit = max(110.0, float(np.nanmax(np.abs(signed))))
    image = ax.imshow(
        masked,
        cmap=DIV_CMAP,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    ax.set_xticks(np.arange(6), np.arange(1, 7))
    ax.set_yticks(np.arange(13), [rf"$q={q}$" for q in range(13)])
    ax.set(
        xlabel="Pair rank by mean absolute share",
        ylabel="Output frequency",
        title="Mean signed harmonic-pair composition",
    )
    for q in range(13):
        for rank in range(6):
            if np.isfinite(signed[q, rank]):
                colour = "white" if abs(signed[q, rank]) > 0.5 * limit else "black"
                ax.text(
                    rank,
                    q,
                    pair_labels[q][rank],
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color=colour,
                )
    fig.colorbar(image, ax=ax, pad=0.025, label="Mean signed contribution\nto output (%)")

    ax = axes[1]
    image = ax.imshow(
        np.clip(spectrum, 1e-6, 1.0), cmap=SEQ_CMAP, norm=LogNorm(1e-6, 1.0), aspect="auto"
    )
    names = _names(pairs, "artery_id", "artery_name")
    names = [
        name.replace("Aortic Root", "Aortic root").replace("Thoracic Aorta", "Thoracic aorta")
        for name in names
    ]
    ax.set_xticks(np.arange(13), np.arange(13))
    ax.set_yticks(np.arange(6), names)
    ax.set(
        xlabel=r"Output frequency, $q$", ylabel="Arterial site", title="Resolved output spectrum"
    )
    fig.colorbar(image, ax=ax, pad=0.025, label=r"$|\widehat{\Phi}_q|/|\widehat{\Phi}_0|$")

    ax = axes[2]
    mean_dominant = dominant.mean(axis=0)
    mean_relative_output = spectrum.mean(axis=0)
    q_values = np.arange(13)
    for q in q_values:
        ax.vlines(q, dominant[:, q].min(), dominant[:, q].max(), color="0.55", linewidth=1.0)
        ax.scatter(np.full(6, q), dominant[:, q], s=12, color="0.72", zorder=2)
    top_pairs = []
    for artery in ARTERY_ORDER:
        artery_rows = pairs[pairs["artery_id"] == artery]
        top_pairs.append(
            [
                tuple(
                    artery_rows[artery_rows["output_frequency"] == q]
                    .sort_values("fraction_of_absolute_pair_sum", ascending=False)
                    .iloc[0][["m", "n"]]
                    .astype(int)
                )
                for q in q_values
            ]
        )
    common_counts = []
    for q in q_values:
        counts = pd.Series([row[q] for row in top_pairs]).value_counts()
        common_counts.append(int(counts.iloc[0]))
    sizes = 35.0 + 55.0 * np.sqrt(np.clip(mean_relative_output, 0.0, 1.0))
    for q, mean_value, size, count in zip(
        q_values, mean_dominant, sizes, common_counts, strict=True
    ):
        filled = count == 6
        ax.scatter(
            q,
            mean_value,
            s=size,
            marker="s",
            facecolors="black" if filled else "white",
            edgecolors="black",
            linewidth=1.0,
            zorder=3,
        )
        if not filled:
            ax.text(q, mean_value + 6.0, f"{count}/6", ha="center", va="bottom", fontsize=5.5)
    ax.axhline(50.0, color="0.75", linestyle="--", linewidth=0.8)
    ax.set_xticks(q_values)
    ax.set(
        xlabel=r"Output frequency, $q$",
        ylabel="Dominant share of\nabsolute pair sum (%)",
        title="Dominant-pair persistence across arteries",
        ylim=(30, 108),
    )
    _clean(ax, True)
    _label(axes)
    paths, record = _save(fig, root, "Figure3", dpi, height, "Figure 3")
    record.update(colormaps=f"{DIV_CMAP},{SEQ_CMAP}", output_frequency_range="0-12")
    return paths, record


def _controls(controls: pd.DataFrame, root: Path, dpi: int):
    height = 66.0
    names = _names(controls, "artery_id", "artery_name")
    names = [
        name.replace("Aortic Root", "Aortic root").replace("Thoracic Aorta", "Thoracic aorta")
        for name in names
    ]
    matched = controls[controls["family"] == "harmonic_removal_rms_matched"].copy()
    matched["harmonic"] = matched["control"].str.extract(r"h([1-6])").astype(int)
    change = (
        100.0
        * matched.pivot(index="artery_id", columns="harmonic", values="fractional_change")
        .loc[ARTERY_ORDER, range(1, 7)]
        .to_numpy()
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(_inch(WIDTH_MM), _inch(height)), constrained_layout=True
    )
    limit = 50.0
    image = axes[0].imshow(
        change,
        cmap=DIV_CMAP,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    axes[0].set_xticks(np.arange(6), [rf"$h={h}$" for h in range(1, 7)])
    axes[0].set_yticks(np.arange(6), names)
    axes[0].set(
        xlabel="Removed pressure harmonic\n(input RMS fixed)",
        ylabel="Arterial site",
        title="Single-harmonic removal",
    )
    for row in range(6):
        for column in range(6):
            value = change[row, column]
            colour = "white" if abs(value) > 0.55 * limit else "black"
            label = f"{value:+.1f}" if value > 0 else f"{value:.1f}"
            axes[0].text(column, row, label, ha="center", va="center", fontsize=5.8, color=colour)
    fig.colorbar(image, ax=axes[0], pad=0.025, label=r"$\Delta\Phi_{\mathrm{rms}}$ (%)")

    phase = controls[controls["control"].str.startswith("phase_random_")]
    common = controls[controls["control"] == "common_phase_pi_over_4"].set_index("artery_id")
    for row, artery in enumerate(ARTERY_ORDER):
        values = 100.0 * phase[phase["artery_id"] == artery]["fractional_change"].to_numpy()
        axes[1].hlines(row, values.min(), values.max(), color="0.55", linewidth=1.0)
        axes[1].scatter(values, np.full(values.size, row), s=13, color="0.72", zorder=2)
        axes[1].scatter(np.median(values), row, marker="s", s=38, color="black", zorder=3)
        common_value = 100.0 * float(common.loc[artery, "fractional_change"])
        axes[1].scatter(
            common_value,
            row,
            marker="D",
            s=55,
            facecolors="white",
            edgecolors="black",
            linewidth=1.0,
            zorder=3,
        )
    axes[1].axvline(0.0, color="0.65", linestyle="--", linewidth=0.9)
    axes[1].set_yticks(np.arange(6), [])
    axes[1].set(
        xlabel=r"$\Delta\Phi_{\mathrm{rms}}$ (%)", title="Phase-only controls", xlim=(-30, 30)
    )
    axes[1].invert_yaxis()
    _clean(axes[1], True)
    _label(axes)
    paths, record = _save(fig, root, "Figure4", dpi, height, "Figure 4")
    record.update(colormap=DIV_CMAP, units="percent")
    return paths, record


def _reduction(predictions: pd.DataFrame, summary: dict, archive, root: Path, dpi: int):
    height = 108.0
    fig, axes = plt.subplots(
        2, 2, figsize=(_inch(WIDTH_MM), _inch(height)), constrained_layout=True
    )
    reduced = summary["reduced_law"]

    singular = np.asarray(archive["singular_values"], dtype=float)
    ratio = singular / singular[0]
    axes[0, 0].semilogy(np.arange(1, len(ratio) + 1), ratio, "o-", color="black", markersize=3.5)
    axes[0, 0].set_xticks(np.arange(1, len(ratio) + 1))
    axes[0, 0].set(
        xlabel=r"Singular mode, $k$",
        ylabel=r"$\sigma_k/\sigma_1$",
        title="Operator singular spectrum",
    )
    axes[0, 0].text(
        0.04,
        0.09,
        f"Rank-one retained energy\n{100.0 * reduced['retained_energy']:.4f}%",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="bottom",
    )
    _clean(axes[0, 0], True)

    folds = (
        pd.DataFrame(reduced["leave_one_out_exponents"])
        .set_index("held_out_artery")
        .loc[ARTERY_ORDER]
        .reset_index()
    )
    y = np.arange(6)
    alpha_deviation = folds["alpha_exponent"] + 2.0
    eta_deviation = folds["eta_exponent"] - 2.0
    for row, (alpha_value, eta_value) in enumerate(
        zip(alpha_deviation, eta_deviation, strict=True)
    ):
        axes[0, 1].hlines(
            row,
            min(alpha_value, eta_value),
            max(alpha_value, eta_value),
            color="0.72",
            linewidth=1.0,
        )
    axes[0, 1].scatter(alpha_deviation, y, s=30, color="black", label=r"$p_\alpha+2$", zorder=3)
    axes[0, 1].scatter(
        eta_deviation,
        y,
        marker="s",
        s=30,
        facecolors="white",
        edgecolors="black",
        linewidth=1.0,
        label=r"$p_\eta-2$",
        zorder=3,
    )
    axes[0, 1].axvline(0.0, color="0.75", linestyle="--", linewidth=0.8)
    names = [
        item.replace("_", " ").capitalize().replace("Aortic root", "Aortic root")
        for item in ARTERY_ORDER
    ]
    axes[0, 1].set_yticks(y, names)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set(
        xlabel="Deviation from canonical exponent",
        ylabel="Held-out artery",
        title="Exponent variation under artery holdout",
    )
    axes[0, 1].legend(frameon=False, loc="lower right")
    _clean(axes[0, 1], True)

    brachial = predictions["held_out_artery"] == "brachial"
    axes[1, 0].scatter(
        predictions.loc[~brachial, "exact_phi_rms"],
        predictions.loc[~brachial, "predicted_phi_rms"],
        s=7,
        color="0.7",
        alpha=0.6,
        label="Other holdouts",
    )
    axes[1, 0].scatter(
        predictions.loc[brachial, "exact_phi_rms"],
        predictions.loc[brachial, "predicted_phi_rms"],
        s=13,
        facecolors="white",
        edgecolors="black",
        linewidth=0.55,
        label="Brachial holdout",
    )
    low = float(min(predictions["exact_phi_rms"].min(), predictions["predicted_phi_rms"].min()))
    high = float(max(predictions["exact_phi_rms"].max(), predictions["predicted_phi_rms"].max()))
    axes[1, 0].plot([low, high], [low, high], color="black")
    axes[1, 0].plot(
        [low, high], [0.9 * low, 0.9 * high], color="0.4", linestyle="--", linewidth=0.8
    )
    axes[1, 0].plot(
        [low, high], [1.1 * low, 1.1 * high], color="0.4", linestyle="--", linewidth=0.8
    )
    axes[1, 0].set(
        xscale="log",
        yscale="log",
        xlabel=r"Full operator, $\Phi_{\mathrm{rms}}$",
        ylabel=r"Reduced law, $\widehat{\Phi}_{\mathrm{rms}}$",
        title="Held-out prediction",
    )
    error_summary = (
        f"Median: {100 * reduced['median_relative_error']:.2f}%\n"
        f"90th percentile: {100 * reduced['p90_relative_error']:.2f}%\n"
        f"Maximum: {100 * reduced['maximum_relative_error']:.2f}%"
    )
    axes[1, 0].text(
        0.04,
        0.96,
        error_summary,
        transform=axes[1, 0].transAxes,
        ha="left",
        va="top",
    )
    axes[1, 0].legend(frameon=False, loc="lower right")
    _clean(axes[1, 0], True)

    grouped = predictions.groupby("held_out_artery")["relative_error"]
    statistics = pd.DataFrame(
        {
            "median": 100.0 * grouped.median(),
            "p90": 100.0 * grouped.quantile(0.9),
            "maximum": 100.0 * grouped.max(),
        }
    ).loc[ARTERY_ORDER]
    for row, values in enumerate(statistics.itertuples()):
        axes[1, 1].hlines(row, values.median, values.maximum, color="0.7", linewidth=1.0)
        axes[1, 1].vlines(values.maximum, row - 0.22, row + 0.22, color="black", linewidth=1.0)
        axes[1, 1].text(
            values.maximum + 0.35,
            row,
            f"{values.maximum:.1f}",
            va="center",
            ha="left",
            fontsize=5.7,
        )
    axes[1, 1].scatter(statistics["median"], y, s=30, color="black", label="Median", zorder=3)
    axes[1, 1].scatter(
        statistics["p90"],
        y,
        marker="s",
        s=30,
        facecolors="white",
        edgecolors="black",
        linewidth=1.0,
        label="90th percentile",
        zorder=3,
    )
    axes[1, 1].axvline(10.0, color="0.75", linestyle="--", linewidth=0.8)
    axes[1, 1].text(10.0, -0.5, "10%", ha="center", va="bottom", color="0.4", fontsize=5.5)
    axes[1, 1].set_yticks(y, names)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set(
        xlabel="Relative prediction error (%)",
        ylabel="Held-out artery",
        title="Held-out prediction error by artery",
        xlim=(0, 18.2),
    )
    axes[1, 1].legend(frameon=False, loc="lower right")
    _clean(axes[1, 1], True)
    _label(axes.ravel())
    return _save(fig, root, "Figure5", dpi, height, "Figure 5")


def _robustness(data: pd.DataFrame, root: Path, dpi: int):
    height = 72.0
    path_order = [
        "reciprocal",
        "beta_low",
        "gamma_low",
        "gamma_only",
        "beta_high_gamma_low",
        "beta_low_gamma_high",
        "delta_low",
        "delta_high",
        "beta_only",
    ]
    path_labels = [
        "Reciprocal",
        r"$\beta/2$",
        r"$\gamma/2$",
        r"$\gamma$ only",
        r"$1.25\beta,0.75\gamma$",
        r"$0.75\beta,1.25\gamma$",
        r"$\delta=0.8$",
        r"$\delta=1.2$",
        r"$\beta$ only",
    ]
    classes = {
        "reciprocal": ("o", True),
        "beta_low": ("s", False),
        "gamma_low": ("s", False),
        "gamma_only": ("s", False),
        "beta_high_gamma_low": ("s", False),
        "beta_low_gamma_high": ("s", False),
        "delta_low": ("^", False),
        "delta_high": ("^", False),
        "beta_only": ("X", True),
    }
    fig = plt.figure(figsize=(_inch(WIDTH_MM), _inch(height)), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(3.35, 1.25))
    ax = fig.add_subplot(grid[0, 0])
    key = fig.add_subplot(grid[0, 1])
    key.axis("off")

    for number, path in enumerate(path_order, start=1):
        selected = data[data["constitutive_path"] == path]
        x = 100.0 * (selected["relative_amplitude_to_reciprocal"].to_numpy() - 1.0)
        y = 100.0 * selected["normalised_shape_relative_l2"].to_numpy()
        if path == "beta_only":
            y = np.zeros_like(y)
        marker, filled = classes[path]
        ax.scatter(x, y, marker=marker, s=18, color="0.85", edgecolors="none", alpha=0.45, zorder=1)
        median_x = float(np.median(x))
        median_y = float(np.median(y))
        ax.scatter(
            median_x,
            median_y,
            marker=marker,
            s=68,
            facecolors="black" if filled else "white",
            edgecolors="black",
            linewidth=1.0,
            zorder=3,
        )
        annotation_offsets = {5: (-16, 9), 8: (8, -19)}
        ax.annotate(
            str(number),
            (median_x, median_y),
            xytext=annotation_offsets.get(number, (7, 6)),
            textcoords="offset points",
            fontsize=5.8,
            fontweight="bold",
        )

    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.axvline(0.0, color="0.75", linestyle="--", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.8)
    ax.set_yticks([0.0, 0.01, 0.1, 1.0, 10.0], ["0", "0.01", "0.1", "1", "10"])
    ax.set(
        xlabel="Amplitude departure from reciprocal path (%)",
        ylabel="Normalised interaction-shape error (%)",
        title="Constitutive shape robustness and amplitude departure",
        xlim=(-105, 32),
        ylim=(-0.002, 20),
    )
    _clean(ax, True)

    class_handles = [
        Line2D(
            [], [], marker="o", linestyle="none", color="black", markersize=5, label="Reciprocal"
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            label="Nonreciprocal",
        ),
        Line2D(
            [],
            [],
            marker="^",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            label="Diagonal shift",
        ),
        Line2D(
            [], [], marker="X", linestyle="none", color="black", markersize=5, label="Null control"
        ),
    ]
    legend = key.legend(
        handles=class_handles,
        title="Constitutive class",
        frameon=False,
        loc="upper left",
        borderaxespad=0.0,
    )
    key.add_artist(legend)
    key.text(
        0.0,
        0.53,
        "Constitutive path",
        transform=key.transAxes,
        fontweight="bold",
        ha="left",
        va="top",
    )
    for index, label in enumerate(path_labels):
        column = 0 if index < 5 else 1
        row = index if index < 5 else index - 5
        key.text(
            0.0 + 0.52 * column,
            0.45 - 0.115 * row,
            f"{index + 1}  {label}",
            transform=key.transAxes,
            ha="left",
            va="top",
            fontsize=5.6,
        )
    paths, record = _save(fig, root, "FigureS1", dpi, height, "Figure S1")
    record.update(x_units="percent", y_scale="symmetric logarithmic", constitutive_paths=9)
    return paths, record


def create_figures(output_root: str | Path, dpi: int = 600) -> list[Path]:
    """Create the five main figures and Supplementary Figure S1."""
    if dpi < 300:
        raise ValueError("dpi must be at least 300 for publication output")
    _style()
    root = Path(output_root).resolve()
    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    atlas = pd.read_csv(root / "artery_atlas.csv")
    crossed = pd.read_csv(root / "crossed_susceptibility.csv")
    controls = pd.read_csv(root / "waveform_controls.csv")
    pairs = pd.read_csv(root / "harmonic_pair_attribution.csv")
    predictions = pd.read_csv(root / "reduced_law_validation.csv")
    robustness = pd.read_csv(root / "constitutive_robustness.csv")
    summary = json.loads((root / "analysis_summary.json").read_text(encoding="utf-8"))
    archive = np.load(root / "operator_archive.npz")
    built = (
        _native(atlas, figure_root, dpi),
        _crossed(crossed, figure_root, dpi),
        _pairs(pairs, figure_root, dpi),
        _controls(controls, figure_root, dpi),
        _reduction(predictions, summary, archive, figure_root, dpi),
        _robustness(robustness, figure_root, dpi),
    )
    archive.close()
    created: list[Path] = []
    records = []
    for paths, record in built:
        created.extend(paths)
        records.append(record)
    manifest = {
        "manuscript_title": (
            "Harmonic interactions shape anisotropy-induced transverse force in arterial blood flow"
        ),
        "journal_family": "Nature Portfolio",
        "target_journal": "Scientific Reports",
        "figure_count": 6,
        "main_figure_count": 5,
        "supplementary_figure_count": 1,
        "common_width_mm": WIDTH_MM,
        "maximum_height_mm": MAX_HEIGHT_MM,
        "font_family": "Arial/Helvetica compatible sans-serif",
        "font_size_pt": FONT_PT,
        "panel_label_size_pt": PANEL_PT,
        "minimum_line_width_pt": LINE_PT,
        "sequential_colormap": SEQ_CMAP,
        "diverging_colormap": DIV_CMAP,
        "background": "white",
        "figures": records,
    }
    (figure_root / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(records).to_csv(figure_root / "figure_manifest.csv", index=False)
    return created
