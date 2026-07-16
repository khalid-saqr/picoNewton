from __future__ import annotations

import numpy as np

from piconewton_v3.model import (
    FluidProperties,
    HydrodynamicConfig,
    SensorConfig,
    V2_ARTERY_CASES,
    WomersleySolver,
    compute_hydrodynamics,
    equilibrium_probability,
    isotropic_validation,
    lamb_work,
    periodic_sensor_solution,
    transition_rates,
)


def test_verified_derivative_matrix_differentiates_r_squared() -> None:
    solver = WomersleySolver(60, "verified")
    assert solver.derivative_polynomial_error() < 1e-10


def test_reproduction_and_verified_modes_are_explicitly_different() -> None:
    verified = WomersleySolver(60, "verified").derivative_polynomial_error()
    reproduction = WomersleySolver(60, "reproduction").derivative_polynomial_error()
    assert verified < 1e-10
    assert reproduction > 1e-2


def test_isotropic_analytic_validation() -> None:
    rows = isotropic_validation((3.0, 8.0, 20.0), radial_order=60)
    assert all(row["passed"] for row in rows)


def test_hydrodynamic_quick_case_is_finite_and_dimensionally_scaled() -> None:
    config = HydrodynamicConfig(
        radial_order=50,
        time_points=128,
        quadrature_nodes=32,
        beta=0.1,
        gamma=0.1,
        delta=1.0,
        mode="verified",
    )
    result = compute_hydrodynamics(V2_ARTERY_CASES[3], config)
    for key in ("force_signed_n", "force_exposure_n", "wall_shear_pa"):
        values = np.asarray(result[key])
        assert values.shape == (128,)
        assert np.isfinite(values).all()
    assert np.asarray(result["force_exposure_n"]).min() >= 0.0
    assert result["max_normalized_backward_residual"] < 1e-12


def test_sensor_local_detailed_balance_and_baseline() -> None:
    sensor = SensorConfig()
    psi = np.linspace(-5.0, 5.0, 101)
    kp, km = transition_rates(psi, sensor)
    expected = sensor.basal_probability / (1 - sensor.basal_probability) * np.exp(psi)
    assert np.allclose(kp / km, expected)
    assert np.isclose(equilibrium_probability(0.0, sensor), sensor.basal_probability)


def test_periodic_sensor_solution_is_bounded_and_closed() -> None:
    sensor = SensorConfig(relaxation_time_s=10.0)
    t = np.arange(512) / 512
    work = 3.0 * np.sin(2 * np.pi * t)
    probability, residual = periodic_sensor_solution(work, 1.2, sensor)
    assert probability.min() >= 0.0
    assert probability.max() <= 1.0
    assert residual < 1e-10


def test_lamb_work_units_at_one_pn_nm() -> None:
    # One pN acting over one nm is 1e-21 J.
    psi = lamb_work(np.array([1e-12]), 1e-9, 310.15)
    expected = 1e-21 / (1.380649e-23 * 310.15)
    assert np.allclose(psi, expected)


def test_all_v2_cases_preserve_six_signed_harmonics() -> None:
    assert len(V2_ARTERY_CASES) == 6
    assert all(len(case.harmonic_coefficients) == 6 for case in V2_ARTERY_CASES)
    assert V2_ARTERY_CASES[2].harmonic_coefficients[3] < 0
    assert V2_ARTERY_CASES[4].harmonic_coefficients[3] < 0
