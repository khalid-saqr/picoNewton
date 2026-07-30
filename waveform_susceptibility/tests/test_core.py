import numpy as np

from piconewton_waveform_susceptibility.core import (
    AnalysisConfig,
    canonical_coefficients,
    critical_anisotropy,
    evaluate_kernel,
)


def test_configuration_defaults_are_valid():
    AnalysisConfig().validate()


def test_canonical_coefficients_are_hermitian():
    frequencies, coefficients = canonical_coefficients([1.0, 0.5j, -0.2])
    assert np.array_equal(frequencies, np.arange(-3, 4))
    for harmonic in range(1, 4):
        assert coefficients[3 - harmonic] == np.conj(coefficients[3 + harmonic])


def test_frequency_mixing_uses_sum_rule():
    frequencies = np.arange(-1, 2)
    kernel = np.ones((3, 3), dtype=complex)
    _, coefficients = canonical_coefficients([2.0])
    output, spectrum, _ = evaluate_kernel(frequencies, kernel, coefficients)
    supported = set(output[np.abs(spectrum) > 1e-14])
    assert supported == {-2, 0, 2}


def test_critical_anisotropy_formula():
    assert np.isclose(critical_anisotropy(4.0, 100.0), 0.2)
