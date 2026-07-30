# ruff: noqa: E501
from __future__ import annotations

from pathlib import Path
import pandas as pd


def _read(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.is_file():
        raise RuntimeError(f"publication table source is missing: {path}")
    return pd.read_csv(path)


def build_publication_tables(step_roots: dict[int, Path], output_root: Path) -> list[Path]:
    table_root = output_root / "tables"
    main = table_root / "main"
    supplementary = table_root / "supplementary"
    main.mkdir(parents=True, exist_ok=True)
    supplementary.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    native = _read(step_roots[6], "native_susceptibility.csv")
    columns = [
        "artery_id", "artery_name", "alpha", "eta", "phi_rms", "phi_peak_abs",
        "outward_duty", "inward_duty", "predicted_rms_at_epsilon_0p1_n",
    ]
    table1 = native[[c for c in columns if c in native.columns]].copy()
    if "predicted_rms_at_epsilon_0p1_n" in table1:
        table1["predicted_rms_at_epsilon_0p1_pn"] = table1.pop("predicted_rms_at_epsilon_0p1_n") * 1e12
    path = main / "table_01_native_susceptibility_atlas.csv"
    table1.to_csv(path, index=False); outputs.append(path)

    validity = _read(step_roots[4], "validity_domains.csv")
    path = main / "table_02_perturbative_validity_domains.csv"
    validity.to_csv(path, index=False); outputs.append(path)

    critical = _read(step_roots[6], "critical_anisotropy.csv")
    path = main / "table_03_critical_anisotropy.csv"
    critical.to_csv(path, index=False); outputs.append(path)

    decomposition = _read(step_roots[7], "crossed_variance_decomposition.csv")
    path = main / "table_04_crossed_effect_decomposition.csv"
    decomposition.to_csv(path, index=False); outputs.append(path)

    selection = _read(step_roots[8], "model_selection.csv")
    path = main / "table_05_reduced_law_selection.csv"
    selection.to_csv(path, index=False); outputs.append(path)

    for source, target in (
        ((5, "kernel_closure.csv"), "table_s01_kernel_closure.csv"),
        ((5, "dominant_pairs.csv"), "table_s02_dominant_pairs.csv"),
        ((7, "native_waveform_controls.csv"), "table_s03_waveform_controls.csv"),
        ((8, "compact_law_family_summary.csv"), "table_s04_reduced_law_family_errors.csv"),
        ((9, "constitutive_path_metrics.csv"), "table_s05_constitutive_robustness.csv"),
        ((9, "resolution_robustness.csv"), "table_s06_resolution_robustness.csv"),
    ):
        frame = _read(step_roots[source[0]], source[1])
        path = supplementary / target
        frame.to_csv(path, index=False); outputs.append(path)
    return outputs
