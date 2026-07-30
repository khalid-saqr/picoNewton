import json
from pathlib import Path

import numpy as np
import pandas as pd

from piconewton_waveform_susceptibility.figures import ARTERY_ORDER, create_figures


def _write_fixture(root: Path) -> None:
    names = {item: item.replace("_", " ").title() for item in ARTERY_ORDER}
    atlas = []
    for index, artery in enumerate(ARTERY_ORDER):
        atlas.append(
            {
                "artery_id": artery,
                "artery_name": names[artery],
                "alpha": 3.0 + 3.0 * index,
                "eta": 0.001 + 0.0002 * index,
                "phi_rms": 1.0 / (index + 1),
                "phi_peak_absolute": 1.4 / (index + 1),
                "outward_duty": 0.45 + 0.01 * index,
                "inward_duty": 0.55 - 0.01 * index,
                "predicted_rms_at_epsilon_0p08_pn": 0.1 / (index + 1),
                "critical_epsilon_1pn_rms": 0.2 + 0.05 * index,
                "critical_epsilon_10pn_rms": 0.6 + 0.1 * index,
            }
        )
    pd.DataFrame(atlas).to_csv(root / "artery_atlas.csv", index=False)

    crossed = []
    for condition, factor in (("reference", 1.0), ("physiological", 1.2)):
        for row, vessel in enumerate(ARTERY_ORDER):
            for column, waveform in enumerate(ARTERY_ORDER):
                crossed.append(
                    {
                        "condition": condition,
                        "vessel_id": vessel,
                        "vessel_name": names[vessel],
                        "waveform_id": waveform,
                        "waveform_name": names[waveform],
                        "phi_rms": factor * (row + 1) * (column + 1) * 1e-4,
                    }
                )
    pd.DataFrame(crossed).to_csv(root / "crossed_susceptibility.csv", index=False)

    controls = []
    for artery in ARTERY_ORDER:
        controls.extend(
            [
                {
                    "artery_id": artery,
                    "artery_name": names[artery],
                    "control": "native",
                    "family": "native",
                    "relative_to_native": 1.0,
                    "fractional_change": 0.0,
                },
                {
                    "artery_id": artery,
                    "artery_name": names[artery],
                    "control": "sign_neutralised",
                    "family": "sign",
                    "relative_to_native": 1.01,
                    "fractional_change": 0.01,
                },
            ]
        )
        for harmonic in range(1, 7):
            ratio = 0.5 + 0.05 * harmonic
            controls.append(
                {
                    "artery_id": artery,
                    "artery_name": names[artery],
                    "control": f"remove_h{harmonic}_rms_matched",
                    "family": "harmonic_removal_rms_matched",
                    "relative_to_native": ratio,
                    "fractional_change": ratio - 1.0,
                }
            )
        for index in range(1, 9):
            ratio = 0.8 + 0.04 * index
            controls.append(
                {
                    "artery_id": artery,
                    "artery_name": names[artery],
                    "control": f"phase_random_{index:02d}",
                    "family": "phase",
                    "relative_to_native": ratio,
                    "fractional_change": ratio - 1.0,
                }
            )
    pd.DataFrame(controls).to_csv(root / "waveform_controls.csv", index=False)

    pairs = []
    for artery in ARTERY_ORDER:
        for output in (0, 1, 2):
            for rank, pair in enumerate(((1, 1), (-1, 2), (2, 2)), start=1):
                pairs.append(
                    {
                        "artery_id": artery,
                        "artery_name": names[artery],
                        "output_frequency": output,
                        "rank": rank,
                        "m": pair[0],
                        "n": pair[1],
                        "fraction_of_absolute_pair_sum": (4 - rank) / 6.0,
                    }
                )
    pd.DataFrame(pairs).to_csv(root / "harmonic_pair_attribution.csv", index=False)

    families = [
        "native",
        "phase_challenge",
        "single_tone",
        "two_tone",
        "spectral_slope",
        "sparse_three_tone",
    ]
    predictions = []
    for artery in ARTERY_ORDER:
        for family_index, family in enumerate(families, start=1):
            for sample in range(1, 4):
                exact = family_index * sample * 1e-3
                relative = 0.01 * (sample - 2)
                predictions.append(
                    {
                        "held_out_artery": artery,
                        "family": family,
                        "exact_phi_rms": exact,
                        "predicted_phi_rms": exact * (1.0 + relative),
                        "relative_error": abs(relative),
                    }
                )
    pd.DataFrame(predictions).to_csv(root / "reduced_law_validation.csv", index=False)

    paths = [
        "reciprocal",
        "beta_low",
        "gamma_low",
        "gamma_only",
        "beta_high_gamma_low",
        "beta_low_gamma_high",
        "delta_low",
        "delta_high",
    ]
    robustness = []
    for artery in ARTERY_ORDER:
        for index, path in enumerate(paths):
            robustness.append(
                {
                    "artery_id": artery,
                    "artery_name": names[artery],
                    "constitutive_path": path,
                    "null_control": False,
                    "normalised_shape_relative_l2": 0.01 * index,
                    "relative_amplitude_to_reciprocal": 1.0 + 0.05 * (index - 3),
                    "maximum_residual": 1e-14 * (index + 1),
                }
            )
    pd.DataFrame(robustness).to_csv(root / "constitutive_robustness.csv", index=False)

    folds = [
        {
            "held_out_artery": artery,
            "alpha_exponent": -2.0 + 0.01 * index,
            "eta_exponent": 2.0 - 0.01 * index,
        }
        for index, artery in enumerate(ARTERY_ORDER)
    ]
    (root / "analysis_summary.json").write_text(
        json.dumps({"reduced_law": {"leave_one_out_exponents": folds}}),
        encoding="utf-8",
    )
    np.savez(root / "operator_archive.npz", singular_values=np.geomspace(1.0, 1e-6, 12))


def test_publication_figure_suite(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    _write_fixture(tmp_path)
    created = create_figures(tmp_path, dpi=300)
    assert len(created) == 18
    manifest = json.loads(
        (tmp_path / "figures/figure_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["figure_count"] == 6
    assert manifest["common_width_mm"] == 180.0
    assert manifest["maximum_height_mm"] <= 170.0
    assert manifest["font_size_pt"] == 7.0
    assert manifest["minimum_line_width_pt"] >= 1.0
    for extension in ("pdf", "svg", "png"):
        assert len(list((tmp_path / "figures").glob(f"figure_[1-6]_*.{extension}"))) == 6
