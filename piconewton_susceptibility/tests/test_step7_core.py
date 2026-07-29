import numpy as np

from piconewton_susceptibility.experiments_core import (
    Step7Config,
    additive_decomposition,
    causal_waveform_families,
    input_rms,
    normalise_input_rms,
)


def test_step7_config_and_family_inventory():
    Step7Config().validate()
    families = causal_waveform_families()
    assert len(families) == 29
    assert {family for _name, family, _coefficients in families} == {
        "single_tone",
        "two_tone",
        "sparse_three_tone",
        "spectral_slope",
    }
    assert all(np.isclose(input_rms(coefficients), 1.0) for _n, _f, coefficients in families)


def test_input_rms_normalisation():
    coefficients = np.array([1.0, 0.5, 0.25, 0.0, 0.0, 0.0], dtype=complex)
    scaled = normalise_input_rms(coefficients, 2.5)
    assert np.isclose(input_rms(scaled), 2.5)


def test_additive_decomposition_closes():
    values = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 7.0], [4.0, 8.0, 12.0]])
    result = additive_decomposition(values)
    assert np.isclose(
        result["vessel_fraction"]
        + result["waveform_fraction"]
        + result["interaction_fraction"],
        1.0,
    )
