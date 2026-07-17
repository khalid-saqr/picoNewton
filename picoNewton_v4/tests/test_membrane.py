import numpy as np
import pytest

from piconewton_v4.membrane import (
    ForceLoadMap,
    MembraneAdmissibleDomain,
    SLSParameters,
    WSSLoadMap,
    harmonic_energy_balance,
    periodic_response,
    step_creep_response,
    validate_passivity,
)


def test_unit_preserving_load_maps():
    assert ForceLoadMap().map(np.array([10e-12]))[0] == pytest.approx(0.1)
    assert WSSLoadMap().map(np.array([1.0]))[0] == pytest.approx(1.0)


def test_step_creep_and_elastic_limit():
    params = SLSParameters()
    time = np.array([0.0, 50.0 * params.creep_time_s])
    strain = step_creep_response(time, stress_step_pa=1.0, params=params)
    assert strain[0] == pytest.approx(1.0 / params.instantaneous_modulus_pa)
    assert strain[-1] == pytest.approx(1.0 / params.relaxed_modulus_pa, abs=1e-12)

    elastic = SLSParameters(
        instantaneous_modulus_pa=2500.0,
        relaxed_modulus_pa=2500.0,
        stress_relaxation_time_s=0.25,
        thickness_m=0.35e-6,
    )
    stress = np.sin(2 * np.pi * np.arange(256) / 256)
    response = periodic_response(stress, dt_s=0.01, params=elastic)
    np.testing.assert_allclose(
        response["equivalent_areal_strain"], stress / 2500.0, atol=1e-15
    )
    assert np.max(response["dissipation_density_w_m3"]) == 0.0


def test_passivity_and_branch_closure():
    params = SLSParameters()
    report = validate_passivity(params)
    assert report["passed"]
    energy = harmonic_energy_balance(
        stress_amplitude_pa=1.0, frequency_hz=1.2, params=params
    )
    assert energy["relative_error"] < 1e-12

    stress = np.cos(2 * np.pi * np.arange(512) / 512)
    response = periodic_response(stress, dt_s=1 / (1.2 * 512), params=params)
    assert np.min(response["dissipation_density_w_m3"]) >= 0.0
    np.testing.assert_allclose(
        response["relaxed_branch_tension_n_m"]
        + response["maxwell_branch_tension_n_m"],
        response["total_applied_tension_n_m"],
        atol=1e-18,
    )


def test_primary_admissible_domain():
    domain = MembraneAdmissibleDomain()
    assert domain.contains(SLSParameters())
    assert not domain.contains(
        SLSParameters(
            instantaneous_modulus_pa=2500.0,
            relaxed_modulus_pa=1000.0,
            stress_relaxation_time_s=2.0,
            thickness_m=0.35e-6,
        )
    )
