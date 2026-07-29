from __future__ import annotations

from dataclasses import replace

import numpy as np

from piconewton_susceptibility.kernel_core import (
    Step5Config,
    canonical_coefficients,
    direct_full_waveform,
    direct_second_order_waveform,
    evaluate_kernel,
    full_force_kernel,
    near_wall_basis,
    reconstruct_spectrum,
    relative_l2,
    second_order_force_kernel,
    unit_full_response,
    unit_perturbation_response,
)
from piconewton_v3 import V2_ARTERY_CASES


def _config() -> Step5Config:
    return Step5Config(
        profile="quick",
        radial_order=40,
        time_points=256,
        quadrature_nodes=64,
        closure_tolerance=1e-9,
    )


def test_canonical_two_sided_coefficients_reconstruct_real_waveform() -> None:
    one_sided = np.array([1.0, -0.4j, 0.3 + 0.2j])
    frequencies, coefficients = canonical_coefficients(one_sided)
    time = np.arange(256) / 256
    two_sided = coefficients @ np.exp(1j * 2 * np.pi * np.outer(frequencies, time))
    harmonics = np.arange(1, 4)
    one_sided_waveform = np.real(
        one_sided @ np.exp(1j * 2 * np.pi * np.outer(harmonics, time))
    )
    assert relative_l2(two_sided.real, one_sided_waveform) < 1e-13
    assert np.max(np.abs(two_sided.imag)) < 1e-13


def test_exact_and_second_order_kernels_close_against_time_products() -> None:
    case = V2_ARTERY_CASES[0]
    config = _config()
    perturbation = unit_perturbation_response(case, config)
    anisotropic = unit_full_response(case, config, 0.07)
    isotropic = unit_full_response(case, config, 0.0)
    _, perturbation = near_wall_basis(case, perturbation, config)
    _, anisotropic = near_wall_basis(case, anisotropic, config)
    _, isotropic = near_wall_basis(case, isotropic, config)
    frequencies, k2 = second_order_force_kernel(case, perturbation)
    _, ka = full_force_kernel(case, anisotropic)
    _, k0 = full_force_kernel(case, isotropic)
    coefficients_one_sided = np.asarray(case.harmonic_coefficients) * np.exp(
        1j * np.array([0.0, 0.2, -0.3, 0.5, -0.7, 0.9])
    )
    _, coefficients = canonical_coefficients(coefficients_one_sided)
    for kernel, direct in (
        (
            k2,
            direct_second_order_waveform(
                case, perturbation, coefficients_one_sided, config.time_points
            ),
        ),
        (
            ka - k0,
            direct_full_waveform(
                case,
                anisotropic,
                isotropic,
                coefficients_one_sided,
                config.time_points,
            ),
        ),
    ):
        q, spectrum, _ = evaluate_kernel(frequencies, kernel, coefficients)
        reconstructed = reconstruct_spectrum(q, spectrum, config.time_points)
        assert relative_l2(reconstructed.real, direct) < 1e-10
        assert np.max(np.abs(reconstructed.imag)) < 1e-10 * np.max(np.abs(direct))


def test_single_tone_selection_rule_is_dc_and_doubling_only() -> None:
    case = replace(V2_ARTERY_CASES[0], harmonic_coefficients=(0, 1, 0, 0, 0, 0))
    config = _config()
    response = unit_perturbation_response(case, config)
    _, response = near_wall_basis(case, response, config)
    frequencies, kernel = second_order_force_kernel(case, response)
    _, coefficients = canonical_coefficients(case.harmonic_coefficients)
    q, spectrum, _ = evaluate_kernel(frequencies, kernel, coefficients)
    support = set(q[np.abs(spectrum) > 1e-12 * np.max(np.abs(spectrum))])
    assert support == {-4, 0, 4}
