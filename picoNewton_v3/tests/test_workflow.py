from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from piconewton_v3.model import FluidProperties, HydrodynamicConfig, SensorConfig, V2_ARTERY_CASES
from piconewton_v3.workflow import (
    evaluate_effect_gates,
    fit_wss_surrogate,
    generate_sobol_design,
    parameter_dominance,
    run_nominal_controls,
    run_parameter_grid,
)


def test_quick_control_and_gate_pipeline() -> None:
    # Two cases and a 3x3 sensor grid keep CI fast while exercising all pathways.
    cases = (V2_ARTERY_CASES[0], V2_ARTERY_CASES[-1])
    config = HydrodynamicConfig(
        radial_order=40,
        time_points=128,
        quadrature_nodes=24,
        beta=0.1,
        gamma=0.1,
        delta=1.0,
        mode="verified",
    )
    controls, waveforms, hydro = run_nominal_controls(cases, config, seed=20260716)
    assert set(controls["control_id"]) >= {
        "C0_zero",
        "C1_WSS",
        "C2_Lamb",
        "C3_parallel",
        "C4_anisotropy_excess",
        "C5_isotropic_Lamb",
        "C6_low_harmonic",
        "C7_phase_scrambled",
        "C8_rms_matched_sinusoid",
        "C9_amplitude_matched_WSS",
        "C11_reverse_direction",
        "C12_magnitude",
    }
    assert controls["minimum"].min() >= -1e-12
    assert controls["maximum"].max() <= 1 + 1e-12
    assert len(waveforms) == len(cases) * config.time_points

    d_values = np.logspace(-10, -8, 3)
    tau_values = np.logspace(-3, 1, 3)
    grid = run_parameter_grid(
        cases,
        hydro,
        FluidProperties(),
        coupling_lengths_m=d_values,
        relaxation_times_s=tau_values,
    )
    assert len(grid) == len(cases) * 9
    surrogate, _ = fit_wss_surrogate(
        cases,
        hydro,
        FluidProperties(),
        SensorConfig(),
        training_ids={cases[0].artery_id},
        held_out_ids={cases[1].artery_id},
    )
    gates = evaluate_effect_gates(grid, surrogate, d_values, tau_values)
    assert set(gates["criterion_id"]) == {f"E{i}" for i in range(1, 9)}
    assert len(parameter_dominance(grid)) == len(cases) * 4


def test_sobol_design_is_power_of_two_and_admissible() -> None:
    design = generate_sobol_design(64, seed=20260716)
    assert len(design) == 64
    margin = design["delta"] - ((design["beta"] + design["gamma"]) / 2) ** 2
    assert (margin > 0).all()


def test_physiological_coverage_checkpoint_resume(tmp_path: Path) -> None:
    import pandas as pd

    from piconewton_v3.workflow import generate_physiological_design, run_physiological_coverage

    project_root = Path(__file__).resolve().parents[1]
    ranges = pd.read_csv(project_root / "data" / "physiological_artery_ranges.csv")
    design = generate_physiological_design(ranges, V2_ARTERY_CASES, 8, seed=20260716)
    profile = HydrodynamicConfig(
        radial_order=30,
        time_points=64,
        quadrature_nodes=12,
        beta=0.1,
        gamma=0.1,
        delta=1.0,
        mode="verified",
    )
    first_summary, first_spectra = run_physiological_coverage(
        design,
        V2_ARTERY_CASES,
        profile,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    second_summary, second_spectra = run_physiological_coverage(
        design,
        V2_ARTERY_CASES,
        profile,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert len(first_summary) == len(second_summary) == 8
    assert len(first_spectra) == len(second_spectra) == 8 * 26
    assert set(first_summary["sample_id"]) == set(second_summary["sample_id"])
