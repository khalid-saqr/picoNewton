"""Immutable effect gates and parameter-dominance summaries."""
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

def _connected_decade_region(
    group: pd.DataFrame,
    coupling_lengths_m: Sequence[float],
    relaxation_times_s: Sequence[float],
    flag: str,
) -> tuple[bool, list[dict[str, float | int]]]:
    d_values = np.asarray(coupling_lengths_m)
    t_values = np.asarray(relaxation_times_s)
    coordinates: set[tuple[int, int]] = set()
    for i, d in enumerate(d_values):
        for j, tau in enumerate(t_values):
            mask = np.isclose(group["coupling_length_m"], d, rtol=1e-12, atol=0.0) & np.isclose(
                group["relaxation_time_s"], tau, rtol=1e-12, atol=0.0
            )
            if mask.any() and bool(group.loc[mask, flag].iloc[0]):
                coordinates.add((i, j))
    components: list[set[tuple[int, int]]] = []
    while coordinates:
        root = coordinates.pop()
        component = {root}
        stack = [root]
        while stack:
            i, j = stack.pop()
            for neighbor in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if neighbor in coordinates:
                    coordinates.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    details: list[dict[str, float | int]] = []
    passed = False
    for component in components:
        ds = [d_values[i] for i, _ in component]
        ts = [t_values[j] for _, j in component]
        d_span = float(max(ds) / min(ds))
        t_span = float(max(ts) / min(ts))
        details.append({"points": len(component), "d_span": d_span, "tau_span": t_span})
        passed |= d_span >= 10.0 or t_span >= 10.0
    details.sort(key=lambda item: int(item["points"]), reverse=True)
    return passed, details[:3]


def evaluate_effect_gates(
    parameter_grid: pd.DataFrame,
    surrogate: pd.DataFrame,
    coupling_lengths_m: Sequence[float],
    relaxation_times_s: Sequence[float],
    physiological_coverage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if physiological_coverage is not None and not physiological_coverage.empty:
        coverage = physiological_coverage.copy()
        coverage["passes_E1_coverage"] = (
            (coverage["effect_parallel_vs_wss"] >= 0.005)
            & (coverage["effect_parallel_vs_wss"] >= 10.0 * NUMERICAL_SENSOR_UNCERTAINTY)
        )
        artery_e1 = coverage.groupby("artery_id")["passes_E1_coverage"].mean()
        e1_arteries = int((artery_e1 >= 0.60).sum())
        e1_fraction = float(coverage["passes_E1_coverage"].mean())
        e1_basis = "physiological coverage"
    else:
        artery_e1 = parameter_grid.groupby("artery_id")["passes_E1_point"].mean()
        e1_arteries = int((artery_e1 >= 0.60).sum())
        e1_fraction = float(parameter_grid["passes_E1_point"].mean())
        e1_basis = "sensor parameter grid"

    e3_details: dict[str, Any] = {}
    for artery_id, group in parameter_grid.groupby("artery_id"):
        e3_details[artery_id] = _connected_decade_region(
            group,
            coupling_lengths_m,
            relaxation_times_s,
            "passes_E1_point",
        )
    e3_pass = sum(bool(value[0]) for value in e3_details.values()) >= 4
    e4_arteries = int(
        (parameter_grid.groupby("artery_id")["passes_E4_point"].mean() >= 0.60).sum()
    )
    e5_arteries = int(
        (parameter_grid.groupby("artery_id")["passes_E5_point"].mean() >= 0.25).sum()
    )
    e6_arteries = int(
        (parameter_grid.groupby("artery_id")["passes_E6_point"].mean() >= 0.25).sum()
    )
    held_out = surrogate[surrogate["split"] == "held_out"]
    e2_pass = bool(not held_out.empty and held_out["passes_E2"].all())
    e7_source = physiological_coverage if physiological_coverage is not None and not physiological_coverage.empty else parameter_grid
    e7_column = "effect_parallel_vs_wss" if "effect_parallel_vs_wss" in e7_source.columns else "effect_parallel_vs_WSS"
    e7_count = int((e7_source.groupby("artery_id")[e7_column].quantile(0.05) >= 0.005).sum())

    rows = [
        {"criterion_id": "E1", "name": "Core Lamb detectability", "passed": bool(e1_arteries >= 4 and e1_fraction >= 0.60), "observed": f"{e1_arteries}/6 arteries at >=60%; overall {e1_fraction:.3f}; basis={e1_basis}"},
        {"criterion_id": "E2", "name": "Held-out WSS nonredundancy", "passed": e2_pass, "observed": json.dumps(held_out.to_dict("records"))},
        {"criterion_id": "E3", "name": "Contiguous parameter support", "passed": bool(e3_pass), "observed": json.dumps(e3_details)},
        {"criterion_id": "E4", "name": "Directional specificity", "passed": bool(e4_arteries >= 4), "observed": f"{e4_arteries}/6 arteries at >=60%"},
        {"criterion_id": "E5", "name": "High-harmonic specificity", "passed": bool(e5_arteries >= 2), "observed": f"{e5_arteries}/6 arteries at >=25%"},
        {"criterion_id": "E6", "name": "Anisotropy-specific detectability", "passed": bool(e6_arteries >= 3), "observed": f"{e6_arteries}/6 arteries at >=25%"},
        {"criterion_id": "E7", "name": "Robust full-range effect", "passed": bool(e7_count >= 4), "observed": f"5th-percentile effect >=0.005 in {e7_count}/6 arteries"},
        {"criterion_id": "E8", "name": "Model-class transparency", "passed": True, "observed": "signed, reversed-direction and magnitude classes are separate"},
    ]
    return pd.DataFrame(rows)


def parameter_dominance(parameter_grid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for artery_name, group in parameter_grid.groupby("artery_name"):
        for parameter in ("coupling_length_m", "relaxation_time_s", "Lambda_RMS", "Omega"):
            values = np.asarray(group[parameter])
            transformed = np.log10(values) if np.all(values > 0) else values
            rho, p_value = spearmanr(transformed, group["effect_parallel_vs_WSS"])
            rows.append({"artery_name": artery_name, "parameter": parameter, "spearman_rho": float(rho), "p_value": float(p_value)})
    return pd.DataFrame(rows)
