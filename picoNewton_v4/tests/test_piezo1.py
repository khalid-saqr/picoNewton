from pathlib import Path

import numpy as np

from piconewton_v4.piezo1 import (
    CLOSED_INITIAL_STATE,
    Piezo1Parameters,
    equilibrated_state,
    generator_matrix,
    propagate_constant,
)
from piconewton_v4.workflow_step5 import validate_committed_cellml


def test_cellml_parameter_transcription_and_source_identity():
    package_root = Path(__file__).resolve().parents[1]
    report = validate_committed_cellml(package_root)
    assert report["passed"], report
    assert report["maximum_parameter_abs_difference"] == 0.0
    assert report["workspace_revision"] == "8465ce0a385cdc40e2f79b001798d600e9f9e4d2"
    assert report["channel_revision"] == "7cbad662ee7cf9ed5e84b776bf3a6811f0ad1267"


def test_detailed_balance_constraints_and_generator_conservation():
    params = Piezo1Parameters()
    assert params.r2 == params.r1 * params.r3 * params.r5 / (params.r4 * params.r6)
    assert params.ce2 == params.ce1 + params.ce3 + params.ce5 - params.ce4 - params.ce6
    assert params.cm2 == -params.cm4
    for pressure in (0.0, 25.0, 70.0):
        for voltage in (-80.0, -40.0, 0.0, 60.0):
            matrix = generator_matrix(pressure, voltage)
            assert np.max(np.abs(np.sum(matrix, axis=0))) < 1e-14


def test_probability_invariant_and_semigroup():
    initial = equilibrated_state(-60.0)
    whole = propagate_constant(initial, 1000.0, 70.0, -60.0)
    split = propagate_constant(initial, 400.0, 70.0, -60.0)
    split = propagate_constant(split, 600.0, 70.0, -60.0)
    assert np.max(np.abs(whole - split)) < 1e-10
    assert np.min(whole) >= -1e-12
    assert abs(np.sum(whole) - 1.0) < 1e-10
    assert np.allclose(np.sum(CLOSED_INITIAL_STATE), 1.0)
