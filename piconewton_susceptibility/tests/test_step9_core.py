import numpy as np
from piconewton_v3 import V2_ARTERY_CASES

from piconewton_susceptibility.reduction_core import kernel_scale
from piconewton_susceptibility.robustness_core import (
    Step9Config,
    derive_general_hierarchy,
    hierarchy_kernel,
    native_eta,
)


def test_general_hierarchy_decomposition_and_beta_only_null():
    case = V2_ARTERY_CASES[0]
    basis = derive_general_hierarchy(case, 50, 1.0)
    eta = native_eta(case)
    reciprocal = hierarchy_kernel(basis, eta, 72, 1.0, 1.0)
    gamma_only = hierarchy_kernel(basis, eta, 72, 0.0, 1.0)
    beta_gamma = reciprocal - gamma_only
    current = hierarchy_kernel(basis, eta, 72, 0.75, 1.25)
    expected = 1.25**2 * gamma_only + 0.75 * 1.25 * beta_gamma
    assert np.linalg.norm(current - expected) / np.linalg.norm(expected) < 1e-12
    beta_only = hierarchy_kernel(basis, eta, 72, 1.0, 0.0)
    assert kernel_scale(beta_only) == 0.0
    assert basis.max_residual < 1e-10


def test_step9_config_rejects_invalid_exact_epsilon():
    config = Step9Config(exact_epsilon=0.2)
    try:
        config.validate()
    except ValueError as error:
        assert "exact_epsilon" in str(error)
    else:
        raise AssertionError("invalid epsilon was accepted")
