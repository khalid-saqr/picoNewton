import pandas as pd

from piconewton_v4.hypotheses import DecisionThresholds, classify_effects


def _effects(current, calcium):
    return pd.DataFrame(
        {
            "artery_id": [f"a{i}" for i in range(6)],
            "hypothesis": ["H3a"] * 6,
            "target": ["signed"] * 6,
            "current_rms_difference_pa": current,
            "calcium_rms_difference_nm": calcium,
        }
    )


def test_primary_decision_uses_current_only():
    effects = _effects(current=[0.1] * 6, calcium=[100.0] * 6)
    result = classify_effects(
        effects,
        DecisionThresholds(current_rms_pa=1.0, calcium_rms_nm=10.0, minimum_arteries=4),
    )
    row = result.iloc[0]
    assert row["primary_endpoint"] == "current"
    assert row["decision"] == "fail"
    assert int(row["passing_arteries"]) == 0
    assert int(row["current_passing_arteries"]) == 0
    assert int(row["calcium_passing_arteries"]) == 6


def test_current_decision_passes_at_predeclared_artery_count():
    effects = _effects(
        current=[2.0, 2.0, 2.0, 2.0, 0.1, 0.1],
        calcium=[0.0] * 6,
    )
    result = classify_effects(
        effects,
        DecisionThresholds(current_rms_pa=1.0, calcium_rms_nm=10.0, minimum_arteries=4),
    )
    row = result.iloc[0]
    assert row["decision"] == "pass"
    assert int(row["passing_arteries"]) == 4


def test_calcium_can_be_screened_separately_without_rescuing_current():
    effects = _effects(current=[0.1] * 6, calcium=[100.0] * 6)
    result = classify_effects(
        effects,
        DecisionThresholds(current_rms_pa=1.0, calcium_rms_nm=10.0, minimum_arteries=4),
        primary_endpoint="calcium",
    )
    row = result.iloc[0]
    assert row["primary_endpoint"] == "calcium"
    assert row["decision"] == "pass"
    assert row["calcium_interpretation"] == "exploratory_uncalibrated"
