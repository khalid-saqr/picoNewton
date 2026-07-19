"""Mechanosensory coupling from hydrodynamic loads to Piezo1 endpoints."""
from __future__ import annotations

from typing import Any
import numpy as np

from .endpoints import EndpointParameters, aggregate_domains, domain_endpoint
from .vector_interface import VectorInterfaceParameters, vector_membrane_state


def simulate_coupled_response(
    *,
    wall_shear_pa: np.ndarray,
    signed_lamb_force_n: np.ndarray,
    lamb_force_exposure_n: np.ndarray,
    dt_s: float,
    interface: VectorInterfaceParameters,
    endpoint: EndpointParameters,
) -> dict[str, Any]:
    """Run the complete vector-resolved membrane-to-Piezo1 calculation.

    The nonnegative exposure signal is passed only to the magnitude-sensitive
    branch and is never interpreted as a signed normal load.
    """
    state = vector_membrane_state(
        wall_shear_pa=wall_shear_pa,
        signed_force_n=signed_lamb_force_n,
        force_exposure_n=lamb_force_exposure_n,
        dt_s=dt_s,
        p=interface,
    )
    apical = domain_endpoint(
        np.asarray(state["apical_pressure_mmhg"]),
        dt_s=dt_s,
        gradmu_mv=interface.gradmu_mv,
        endpoint=endpoint,
    )
    junctional = domain_endpoint(
        np.asarray(state["junctional_pressure_mmhg"]),
        dt_s=dt_s,
        gradmu_mv=interface.gradmu_mv,
        endpoint=endpoint,
    )
    aggregate = aggregate_domains(
        apical,
        junctional,
        apical_fraction=interface.apical_channel_fraction,
    )
    return {
        "membrane": state,
        "apical": apical,
        "junctional": junctional,
        "aggregate": aggregate,
    }


__all__ = ["simulate_coupled_response"]
