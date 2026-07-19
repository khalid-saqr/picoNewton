"""Predeclared hypothesis-screening utilities.

Thresholds are supplied by configuration rather than embedded as conclusions.
This module classifies workflow effect tables without modifying the model.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pandas as pd


@dataclass(frozen=True)
class DecisionThresholds:
    current_rms_pa: float
    calcium_rms_nm: float
    minimum_arteries: int = 4

    def validate(self) -> None:
        if self.current_rms_pa <= 0 or self.calcium_rms_nm <= 0:
            raise ValueError("positive current and calcium thresholds are required")
        if not 1 <= self.minimum_arteries <= 6:
            raise ValueError("minimum_arteries must lie in [1, 6]")


def classify_effects(
    effects: pd.DataFrame,
    thresholds: DecisionThresholds,
) -> pd.DataFrame:
    """Classify each hypothesis/target pair against explicit endpoint limits."""
    thresholds.validate()
    required = {
        "artery_id",
        "hypothesis",
        "target",
        "current_rms_difference_pa",
        "calcium_rms_difference_nm",
    }
    missing = required.difference(effects.columns)
    if missing:
        raise ValueError(f"effect table is missing columns: {sorted(missing)}")

    frame = effects.copy()
    frame["passes_current"] = (
        frame["current_rms_difference_pa"] >= thresholds.current_rms_pa
    )
    frame["passes_calcium"] = (
        frame["calcium_rms_difference_nm"] >= thresholds.calcium_rms_nm
    )
    frame["passes_either_endpoint"] = frame["passes_current"] | frame["passes_calcium"]

    rows: list[dict[str, object]] = []
    for (hypothesis, target), group in frame.groupby(["hypothesis", "target"], sort=True):
        passing = int(group["passes_either_endpoint"].sum())
        rows.append(
            {
                "hypothesis": hypothesis,
                "target": target,
                "passing_arteries": passing,
                "required_arteries": thresholds.minimum_arteries,
                "decision": "pass" if passing >= thresholds.minimum_arteries else "fail",
                "current_threshold_pa": thresholds.current_rms_pa,
                "calcium_threshold_nm": thresholds.calcium_rms_nm,
            }
        )
    return pd.DataFrame(rows)


def write_decisions(
    effects_csv: Path,
    output_csv: Path,
    thresholds: DecisionThresholds,
    metadata_json: Path | None = None,
) -> pd.DataFrame:
    decisions = classify_effects(pd.read_csv(effects_csv), thresholds)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output_csv, index=False)
    if metadata_json is not None:
        metadata_json.write_text(
            json.dumps({"thresholds": asdict(thresholds)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return decisions


__all__ = ["DecisionThresholds", "classify_effects", "write_decisions"]
