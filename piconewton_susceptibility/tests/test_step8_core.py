import numpy as np

from piconewton_susceptibility.reduction_core import (
    Step8Config,
    canonical_coefficients,
    susceptibility_from_kernel,
    truncated_kernel,
    waveform_catalog,
)


def test_step8_catalog_and_config():
    config = Step8Config()
    config.validate()
    catalogue = waveform_catalog(config)
    families = {row["family"] for row in catalogue}
    assert families == {
        "native",
        "single_tone",
        "two_tone",
        "sparse_three_tone",
        "spectral_slope",
        "phase_challenge",
    }
    assert len(catalogue) == 89


def test_canonical_coefficients_conjugacy():
    values = canonical_coefficients([1, 2j, 3, 4, 5, 6])
    assert values.shape == (12,)
    assert np.allclose(values[:6], np.conj(values[6:][::-1]))


def test_rank_one_kernel_is_exactly_retained():
    left = np.arange(1, 13, dtype=float) + 1j * np.arange(12)
    right = np.arange(12, 0, -1, dtype=float) - 0.5j
    kernel = np.outer(left, right)
    reduced, singular_values, retained = truncated_kernel(kernel, 1)
    assert retained > 1 - 1e-12
    assert np.linalg.norm(reduced - kernel) / np.linalg.norm(kernel) < 1e-12
    assert susceptibility_from_kernel(reduced, np.ones(6)) > 0
    assert singular_values[0] > 0
