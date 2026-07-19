import pandas as pd
from piconewton_v4.hypotheses import DecisionThresholds, classify_effects


def test_hypothesis_classification_uses_explicit_thresholds():
    effects = pd.DataFrame(
        {
            "artery_id": [f"a{i}" for i in range(6)],
            "hypothesis": ["H3a"] * 6,
            "target": ["signed"] * 6,
            "current_rms_difference_pa": [2.0, 2.0, 2.0, 2.0, 0.1, 0.1],
            "calcium_rms_difference_nm": [0.0] * 6,
        }
    )
    result = classify_effects(
        effects,
        DecisionThresholds(current_rms_pa=1.0, calcium_rms_nm=10.0, minimum_arteries=4),
    )
    assert result.iloc[0]["decision"] == "pass"
    assert int(result.iloc[0]["passing_arteries"]) == 4
