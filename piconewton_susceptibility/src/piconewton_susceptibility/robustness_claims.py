from __future__ import annotations

from typing import Any

def claim_lock(law: dict[str, Any], passed: bool) -> dict[str, Any]:
    return {
        "status": "locked" if passed else "revision_required",
        "selected_law": {
            "rank": 1,
            "prefactor": float(law["prefactor"]),
            "alpha_exponent": float(law["alpha_exponent"]),
            "eta_exponent": float(law["eta_exponent"]),
        },
        "permitted_primary_claim": (
            "Within the straight, rigid, axisymmetric six-harmonic model and the "
            "tested alpha-eta domain, reciprocal weak constitutive anisotropy produces "
            "a signed transverse near-wall force susceptibility represented by a frozen "
            "rank-one, phase-aware waveform functional with vessel scaling approximately "
            "alpha^-2 eta^2."
        ),
        "permitted_secondary_claims": [
            (
                "The normalised harmonic-interaction shape remains predictive under the "
                "declared moderate nonreciprocal beta-gamma and delta perturbations."
            ),
            (
                "The reciprocal amplitude law is stable to the declared radial, "
                "quadrature and near-wall-boundary perturbations."
            ),
            (
                "Azimuthal generation requires gamma at leading order; beta alone "
                "produces no second-order excess."
            ),
        ],
        "required_qualifier": (
            "Amplitude universality is restricted to beta=gamma on the reciprocal path "
            "with delta=1. Nonreciprocal and delta-shifted tensors require a separate "
            "constitutive amplitude factor and are not predicted by the frozen reciprocal "
            "prefactor."
        ),
        "prohibited_claims": [
            "constitutive-universal amplitude law",
            "biological activation threshold",
            "in-vivo force prediction",
            "traction equivalence",
            "total radial acceleration equivalence",
            "measured phase generality",
            "extension beyond six pressure harmonics",
            "causal disease prediction",
        ],
        "allowed_next_step": 10 if passed else None,
    }


