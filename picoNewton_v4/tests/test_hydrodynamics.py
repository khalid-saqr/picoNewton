from pathlib import Path

import numpy as np

from piconewton_v4.hydrodynamics import (
    WomersleySolver,
    compute_decomposition,
    isotropic_validation,
)
from piconewton_v4.types import HydrodynamicConfig, load_artery_cases


ROOT = Path(__file__).resolve().parents[1]


def test_spectral_derivative_and_isotropic_solution():
    solver = WomersleySolver(80)
    assert solver.derivative_polynomial_error() < 1e-11
    rows = isotropic_validation((3.0, 8.0, 20.0), radial_order=80)
    assert all(row["passed"] for row in rows)
    assert max(row["linf_error"] for row in rows) < 1e-8


def test_six_artery_ground_truth_alpha():
    cases = load_artery_cases(ROOT / "data" / "ground_truth_arteries.csv")
    assert len(cases) == 6
    config = HydrodynamicConfig(radial_order=40, time_points=128, near_wall_nodes=32)
    for case in cases:
        result = compute_decomposition(case, config)
        diag = result["anisotropic_diagnostics"]
        assert abs(diag["alpha"] - diag["alpha_reference"]) < 0.02
        assert diag["max_normalized_backward_residual"] < 1e-12
        assert diag["gromeka_lamb_closure_relative_error"] < 1e-8


def test_force_wss_and_decomposition_are_distinct():
    case = load_artery_cases(ROOT / "data" / "ground_truth_arteries.csv")[0]
    result = compute_decomposition(
        case,
        HydrodynamicConfig(radial_order=40, time_points=128, near_wall_nodes=32),
    )
    wss = np.asarray(result["wss_anisotropic_pa"])
    signed = np.asarray(result["force_signed_anisotropic_n"])
    exposure = np.asarray(result["force_exposure_anisotropic_n"])
    delta = np.asarray(result["force_signed_anisotropy_increment_n"])
    assert wss.shape == signed.shape == exposure.shape == delta.shape
    assert np.all(exposure >= 0.0)
    assert np.max(np.abs(signed)) > 0.0
    assert np.max(np.abs(delta)) > 0.0
    assert not np.allclose(signed / np.max(np.abs(signed)), wss / np.max(np.abs(wss)))
