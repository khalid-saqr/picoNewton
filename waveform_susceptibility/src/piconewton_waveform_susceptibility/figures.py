from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

ARTERY_ORDER = [
    "aortic_root", "thoracic_aorta", "femoral", "carotid", "iliac", "brachial"
]
WIDTH_MM = 180.0
MAX_HEIGHT_MM = 170.0
FONT_PT = 7.0
PANEL_PT = 8.0
LINE_PT = 1.0
SEQ_CMAP = "cividis"
DIV_CMAP = "RdBu_r"


def _inch(mm: float) -> float:
    return mm / 25.4


def _style() -> None:
    mpl.rcParams.update({
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
    })


def _label(axes: Iterable[plt.Axes]) -> None:
    for letter, axis in zip("abcdefghijklmnopqrstuvwxyz", axes, strict=False):
        axis.text(-0.14, 1.05, letter, transform=axis.transAxes,
                  fontsize=PANEL_PT, fontweight="bold", va="bottom")


def _clean(axis: plt.Axes, grid: bool = False) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(True, linewidth=0.45, alpha=0.25)
        axis.set_axisbelow(True)


def _save(fig: plt.Figure, root: Path, stem: str, dpi: int, height: float):
    paths = []
    for suffix in ("pdf", "svg", "png"):
        path = root / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi if suffix == "png" else None,
                    bbox_inches="tight", pad_inches=0.03)
        paths.append(path)
    plt.close(fig)
    return paths, {
        "figure": stem,
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
    return (frame.drop_duplicates(key).set_index(key).loc[ARTERY_ORDER, name].tolist())


def _palette() -> dict[str, np.ndarray]:
    values = plt.get_cmap(SEQ_CMAP)(np.linspace(0.08, 0.92, 6))
    return dict(zip(ARTERY_ORDER, values, strict=True))


def _native(atlas: pd.DataFrame, root: Path, dpi: int):
    height = 132.0
    fig, axes = plt.subplots(2, 2, figsize=(_inch(WIDTH_MM), _inch(height)),
                             constrained_layout=True)
    atlas = _ordered(atlas, "artery_id")
    names = atlas["artery_name"].tolist()
    colours = [_palette()[item] for item in atlas["artery_id"]]

    ax = axes[0, 0]
    ax.plot(atlas["alpha"], atlas["phi_rms"], color="0.25")
    ax.scatter(atlas["alpha"], atlas["phi_rms"], c=colours, s=28)
    ax.set(xscale="log", yscale="log", xlabel=r"Womersley number, $\alpha$",
           ylabel=r"RMS susceptibility, $\Phi_{2,\mathrm{rms}}$")
    _clean(ax, True)

    ax = axes[0, 1]
    ax.bar(np.arange(6), atlas["predicted_rms_at_epsilon_0p08_pn"], color=colours)
    ax.set_xticks(np.arange(6), names, rotation=35, ha="right")
    ax.set_ylabel(r"RMS excess force at $\varepsilon=0.08$ (pN)")
    _clean(ax, True)

    x = np.arange(6)
    outward = 100 * atlas["outward_duty"]
    inward = 100 * atlas["inward_duty"]
    axes[1, 0].bar(x, outward, color=plt.get_cmap(SEQ_CMAP)(0.75), label="Outward")
    axes[1, 0].bar(x, inward, bottom=outward,
                   color=plt.get_cmap(SEQ_CMAP)(0.25), label="Inward")
    axes[1, 0].set_xticks(x, names, rotation=35, ha="right")
    axes[1, 0].set(ylabel="Directional duty (%)", ylim=(0, 100))
    axes[1, 0].legend(frameon=False, ncols=2)
    _clean(axes[1, 0])

    ax = axes[1, 1]
    ax.scatter(atlas["alpha"], atlas["critical_epsilon_1pn_rms"],
               c=colours, s=28, label="1 pN reference")
    ax.scatter(atlas["alpha"], atlas["critical_epsilon_10pn_rms"],
               edgecolors=colours, facecolors="none", marker="s", s=24,
               label="10 pN reference")
    ax.axhline(0.08, color="0.25", linestyle="--", label="Validated limit")
    ax.set(xscale="log", yscale="log", xlabel=r"Womersley number, $\alpha$",
           ylabel=r"Formal anisotropy, $\varepsilon_{\mathrm{crit}}$")
    ax.legend(frameon=False)
    _clean(ax, True)
    _label(axes.ravel())
    return _save(fig, root, "figure_1_native_susceptibility", dpi, height)


def _crossed(crossed: pd.DataFrame, root: Path, dpi: int):
    height = 92.0
    matrices = []
    for condition in ("reference", "physiological"):
        selected = crossed[crossed["condition"] == condition]
        matrix = selected.pivot(index="vessel_id", columns="waveform_id",
                                values="phi_rms").loc[ARTERY_ORDER, ARTERY_ORDER]
        matrices.append(matrix.to_numpy())
    names_v = _names(crossed, "vessel_id", "vessel_name")
    names_w = _names(crossed, "waveform_id", "waveform_name")
    values = np.concatenate([item.ravel() for item in matrices])
    positive = values[values > 0]
    norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    fig, axes = plt.subplots(1, 2, figsize=(_inch(WIDTH_MM), _inch(height)),
                             constrained_layout=True)
    for ax, matrix, title in zip(axes, matrices,
                                 ("Common near-wall ratio", "Native near-wall ratio"),
                                 strict=True):
        image = ax.imshow(matrix, cmap=SEQ_CMAP, norm=norm, aspect="equal")
        ax.set_xticks(np.arange(6), names_w, rotation=38, ha="right")
        ax.set_yticks(np.arange(6), names_v)
        ax.set(xlabel="Transferred pressure waveform", title=title)
    axes[0].set_ylabel("Vessel response")
    fig.colorbar(image, ax=axes, shrink=0.86, pad=0.03,
                 label=r"RMS susceptibility, $\Phi_{2,\mathrm{rms}}$")
    _label(axes)
    paths, record = _save(fig, root, "figure_2_crossed_susceptibility", dpi, height)
    record.update(colormap=SEQ_CMAP, normalisation="shared logarithmic")
    return paths, record


def _controls(controls: pd.DataFrame, root: Path, dpi: int):
    height = 92.0
    names = _names(controls, "artery_id", "artery_name")
    matched = controls[controls["family"] == "harmonic_removal_rms_matched"].copy()
    matched["harmonic"] = matched["control"].str.extract(r"h([1-6])").astype(int)
    matrix = matched.pivot(index="artery_id", columns="harmonic",
                           values="relative_to_native").loc[ARTERY_ORDER, range(1, 7)]
    reduction = 100 * (1 - matrix.to_numpy())
    limit = float(np.max(np.abs(reduction)))
    fig, axes = plt.subplots(1, 3, figsize=(_inch(WIDTH_MM), _inch(height)),
                             constrained_layout=True)
    image = axes[0].imshow(reduction, cmap=DIV_CMAP,
                           norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                           aspect="auto")
    axes[0].set_xticks(np.arange(6), [f"h={item}" for item in range(1, 7)])
    axes[0].set_yticks(np.arange(6), names)
    axes[0].set(xlabel="Removed harmonic", ylabel="Artery")
    fig.colorbar(image, ax=axes[0], pad=0.02, label="RMS change (%)")

    fundamental = _ordered(matched[matched["harmonic"] == 1], "artery_id")
    axes[1].bar(np.arange(6), 100 * (1 - fundamental["relative_to_native"]),
                color=[_palette()[item] for item in ARTERY_ORDER])
    axes[1].set_xticks(np.arange(6), names, rotation=38, ha="right")
    axes[1].set_ylabel("Fundamental-removal reduction (%)")
    _clean(axes[1], True)

    phase = controls[controls["control"].str.startswith("phase_random_")]
    groups = [100 * (phase[phase["artery_id"] == item]["relative_to_native"] - 1)
              for item in ARTERY_ORDER]
    axes[2].boxplot(groups, labels=names, showfliers=False)
    axes[2].axhline(0, color="0.25", linewidth=0.8)
    axes[2].tick_params(axis="x", rotation=38)
    axes[2].set_ylabel("Random-phase change (%)")
    _clean(axes[2], True)
    _label(axes)
    paths, record = _save(fig, root, "figure_3_waveform_controls", dpi, height)
    record.update(colormaps=f"{DIV_CMAP},{SEQ_CMAP}", units="percent")
    return paths, record


def _pairs(pairs: pd.DataFrame, root: Path, dpi: int):
    height = 96.0
    names = _names(pairs, "artery_id", "artery_name")
    fig, axes = plt.subplots(1, 3, figsize=(_inch(WIDTH_MM), _inch(height)),
                             constrained_layout=True)
    for ax, output in zip(axes, (0, 1, 2), strict=True):
        selected = pairs[(pairs["output_frequency"] == output) & (pairs["rank"] <= 3)].copy()
        selected["pair"] = [f"({row.m},{row.n})" for row in selected.itertuples()]
        order = (selected.groupby("pair")["fraction_of_absolute_pair_sum"].mean()
                 .sort_values(ascending=False).index[:5])
        matrix = (selected[selected["pair"].isin(order)]
                  .pivot_table(index="artery_id", columns="pair",
                               values="fraction_of_absolute_pair_sum", fill_value=0)
                  .reindex(index=ARTERY_ORDER, columns=order, fill_value=0))
        image = ax.imshow(100 * matrix, cmap=SEQ_CMAP,
                          norm=Normalize(vmin=0, vmax=100), aspect="auto")
        ax.set_xticks(np.arange(len(order)), order, rotation=38, ha="right")
        ax.set_yticks(np.arange(6), names if output == 0 else [])
        ax.set(xlabel=r"Harmonic pair $(m,n)$", title=f"Output q={output}")
    axes[0].set_ylabel("Artery")
    fig.colorbar(image, ax=axes, shrink=0.86, pad=0.03,
                 label="Pairwise absolute share (%)")
    _label(axes)
    paths, record = _save(fig, root, "figure_4_harmonic_interactions", dpi, height)
    record.update(colormap=SEQ_CMAP, normalisation="shared linear 0-100 percent")
    return paths, record


def _reduction(predictions: pd.DataFrame, summary: dict, archive, root: Path, dpi: int):
    height = 132.0
    fig, axes = plt.subplots(2, 2, figsize=(_inch(WIDTH_MM), _inch(height)),
                             constrained_layout=True)
    families = sorted(predictions["family"].unique())
    colours = plt.get_cmap(SEQ_CMAP)(np.linspace(0.08, 0.92, len(families)))
    for family, colour in zip(families, colours, strict=True):
        selected = predictions[predictions["family"] == family]
        axes[0, 0].scatter(selected["exact_phi_rms"], selected["predicted_phi_rms"],
                           s=8, alpha=0.55, color=colour,
                           label=family.replace("_", " "))
    low = float(min(predictions["exact_phi_rms"].min(),
                    predictions["predicted_phi_rms"].min()))
    high = float(max(predictions["exact_phi_rms"].max(),
                     predictions["predicted_phi_rms"].max()))
    axes[0, 0].plot([low, high], [low, high], color="0.2", linestyle="--")
    axes[0, 0].set(xscale="log", yscale="log",
                   xlabel=r"Full-operator $\Phi_{2,\mathrm{rms}}$",
                   ylabel=r"Rank-one $\widehat{\Phi}_{2,\mathrm{rms}}$")
    _clean(axes[0, 0], True)

    errors = [100 * predictions[predictions["family"] == item]["relative_error"]
              for item in families]
    axes[0, 1].boxplot(errors, labels=[item.replace("_", " ") for item in families],
                       showfliers=False)
    axes[0, 1].tick_params(axis="x", rotation=38)
    axes[0, 1].set_ylabel("Relative prediction error (%)")
    _clean(axes[0, 1], True)

    folds = pd.DataFrame(summary["reduced_law"]["leave_one_out_exponents"])
    x = np.arange(len(folds))
    axes[1, 0].plot(x, folds["alpha_exponent"], "o-", label=r"$p_\alpha$")
    axes[1, 0].plot(x, folds["eta_exponent"], "s-", label=r"$p_\eta$")
    axes[1, 0].axhline(-2, color="0.4", linestyle="--", linewidth=0.8)
    axes[1, 0].axhline(2, color="0.4", linestyle="--", linewidth=0.8)
    axes[1, 0].set_xticks(x, folds["held_out_artery"], rotation=38, ha="right")
    axes[1, 0].set_ylabel("Fitted exponent")
    axes[1, 0].legend(frameon=False, ncols=2)
    _clean(axes[1, 0], True)

    singular = np.asarray(archive["singular_values"], dtype=float)
    energy = singular**2 / np.sum(singular**2)
    axes[1, 1].semilogy(np.arange(1, len(energy) + 1), energy, "o-",
                         color=plt.get_cmap(SEQ_CMAP)(0.7))
    axes[1, 1].set(xlabel="Interaction-kernel mode",
                   ylabel="Retained energy fraction")
    _clean(axes[1, 1], True)
    _label(axes.ravel())
    return _save(fig, root, "figure_5_reduced_law", dpi, height)


def _robustness(data: pd.DataFrame, root: Path, dpi: int):
    height = 100.0
    data = data[~data["null_control"]].copy()
    paths = list(dict.fromkeys(data["constitutive_path"]))
    names = _names(data, "artery_id", "artery_name")
    shape = (data.pivot(index="constitutive_path", columns="artery_id",
                        values="normalised_shape_relative_l2")
             .reindex(index=paths, columns=ARTERY_ORDER).to_numpy())
    amplitude = (data.pivot(index="constitutive_path", columns="artery_id",
                            values="relative_amplitude_to_reciprocal")
                 .reindex(index=paths, columns=ARTERY_ORDER).to_numpy())
    fig, axes = plt.subplots(1, 3, figsize=(_inch(WIDTH_MM), _inch(height)),
                             constrained_layout=True)
    image = axes[0].imshow(100 * shape, cmap=SEQ_CMAP, aspect="auto")
    axes[0].set_xticks(np.arange(6), names, rotation=38, ha="right")
    axes[0].set_yticks(np.arange(len(paths)), [item.replace("_", " ") for item in paths])
    axes[0].set_ylabel("Constitutive path")
    fig.colorbar(image, ax=axes[0], pad=0.02, label="Shape error (%)")

    departure = float(np.max(np.abs(amplitude - 1)))
    image = axes[1].imshow(amplitude, cmap=DIV_CMAP,
                           norm=TwoSlopeNorm(vmin=1 - departure, vcenter=1,
                                             vmax=1 + departure), aspect="auto")
    axes[1].set_xticks(np.arange(6), names, rotation=38, ha="right")
    axes[1].set_yticks(np.arange(len(paths)), [])
    fig.colorbar(image, ax=axes[1], pad=0.02, label="Amplitude ratio")

    residual = data.groupby("constitutive_path", sort=False)["maximum_residual"].max()
    axes[2].semilogy(np.arange(len(residual)), residual, "o-",
                     color=plt.get_cmap(SEQ_CMAP)(0.7))
    axes[2].set_xticks(np.arange(len(residual)),
                       [item.replace("_", " ") for item in residual.index],
                       rotation=38, ha="right")
    axes[2].set_ylabel("Maximum normalised residual")
    _clean(axes[2], True)
    _label(axes)
    paths_out, record = _save(fig, root, "figure_6_constitutive_robustness",
                              dpi, height)
    record.update(colormaps=f"{SEQ_CMAP},{DIV_CMAP}")
    return paths_out, record


def create_figures(output_root: str | Path, dpi: int = 600) -> list[Path]:
    """Create six full-width, multi-panel publication figures."""
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
        _controls(controls, figure_root, dpi),
        _pairs(pairs, figure_root, dpi),
        _reduction(predictions, summary, archive, figure_root, dpi),
        _robustness(robustness, figure_root, dpi),
    )
    archive.close()
    created, records = [], []
    for paths, record in built:
        created.extend(paths)
        records.append(record)
    manifest = {
        "journal_family": "Nature Portfolio",
        "target_journal": "Scientific Reports",
        "figure_count": 6,
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
