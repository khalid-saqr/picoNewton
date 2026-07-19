import numpy as np
import pandas as pd

from piconewton_v4.hypotheses import DecisionThresholds
from piconewton_v4.scientific_study import (
    build_completion_assessment,
    build_domain_degeneracy_audit,
    build_endpoint_decisions,
    build_pressure_clipping_audit,
)


def test_current_decisions_do_not_use_calcium_or_rule():
    effects = pd.DataFrame(
        {
            "artery_id": [f"a{i}" for i in range(6)],
            "hypothesis": ["H3a"] * 6,
            "target": ["signed"] * 6,
            "current_rms_difference_pa": [0.2] * 6,
            "calcium_rms_difference_nm": [100.0] * 6,
        }
    )
    current, calcium = build_endpoint_decisions(
        effects,
        DecisionThresholds(3.3, 10.0, 4),
    )
    assert current.iloc[0]["decision"] == "fail"
    assert calcium.iloc[0]["exploratory_calcium_signal"] == "pass"


def test_signed_exposure_degeneracy_is_explicit():
    summary = pd.DataFrame(
        {
            "artery_id": ["aortic_root", "aortic_root"],
            "pathway": ["signed", "exposure"],
            "spatial_current_polarity_index": [0.2, -0.2],
        }
    )
    signal = np.linspace(0.0, 1.0, 8)
    waveforms = {
        "aortic_root_signed_P_Open": signal,
        "aortic_root_exposure_P_Open": signal.copy(),
        "aortic_root_signed_current_pA": signal,
        "aortic_root_exposure_current_pA": signal.copy(),
        "aortic_root_signed_calcium_nm": signal,
        "aortic_root_exposure_calcium_nm": signal.copy(),
    }
    audit = build_domain_degeneracy_audit(summary, waveforms)
    assert bool(audit.iloc[0]["aggregate_degenerate"])
    assert audit.iloc[0]["interpretation"] == "spatially_distinct_but_aggregate_indistinguishable"


def test_pressure_clipping_audit_and_negative_outcome():
    waveforms = {
        "aortic_root_zero_apical_pressure_mmhg": np.array([1.0, 2.0]),
        "aortic_root_zero_junctional_pressure_mmhg": np.array([1.0, 2.0]),
        "aortic_root_vector_apical_pressure_mmhg": np.array([60.0, 70.0]),
        "aortic_root_vector_junctional_pressure_mmhg": np.array([20.0, 30.0]),
    }
    clipping = build_pressure_clipping_audit(waveforms, maximum_pressure_mmhg=70.0)
    assert clipping["clipping_present"].any()

    assessment = build_completion_assessment(
        workflow_manifest={
            "status": "passed_structural_validation",
            "calibration_audit": {"complete": False},
            "endpoint_reference": {"calibration_status": "literature_proxy"},
        },
        current_decisions=pd.DataFrame({"decision": ["fail"]}),
        degeneracy=pd.DataFrame({"aggregate_degenerate": [True]}),
        clipping=clipping,
        hydrodynamic_archive={"array_count": 10},
    )
    assert assessment["study_outcome"] == "negative_under_current_parameterization"
    assert assessment["claims_enabled"] is False
