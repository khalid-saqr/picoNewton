"""Predeclared hypothesis-screening utilities.

Current is the primary endpoint for mechanosensory decisions. The calcium-scale
output remains available as an explicitly exploratory diagnostic until it is
independently calibrated against endothelial measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

import pandas as pd


PrimaryEndpoint = Literal["current", "calcium"]


@dataclass(frozen=True)
class DecisionThresholds:
    """Frozen detection thresholds used after the model has run."""

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
    *,
    primary_endpoint: PrimaryEndpoint = "current",
) -> pd.DataFrame:
    """Classify each hypothesis/target pair using one declared primary endpoint.

    ``current`` is the default and publication-facing endpoint. Calcium can be
    screened separately, but it is never combined with current through an OR
    rule. The returned table always reports both endpoint counts so the reason
    for every decision remains auditable.
    """

    thresholds.validate()
    if primary_endpoint not in {"current", "calcium"}:
        raise ValueError("primary_endpoint must be 'current' or 'calcium'")

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

    rows: list[dict[str, object]] = []
    for (hypothesis, target), group in frame.groupby(
        ["hypothesis", "target"], sort=True
    ):
        current_passing = int(group["passes_current"].sum())
        calcium_passing = int(group["passes_calcium"].sum())
        primary_passing = (
            current_passing if primary_endpoint == "current" else calcium_passing
        )
        rows.append(
            {
                "hypothesis": hypothesis,
                "target": target,
                "passing_arteries": primary_passing,
                "current_passing_arteries": current_passing,
                "calcium_passing_arteries": calcium_passing,
                "required_arteries": thresholds.minimum_arteries,
                "primary_endpoint": primary_endpoint,
                "decision": (
                    "pass"
                    if primary_passing >= thresholds.minimum_arteries
                    else "fail"
                ),
                "current_threshold_pa": thresholds.current_rms_pa,
                "calcium_threshold_nm": thresholds.calcium_rms_nm,
                "calcium_interpretation": "exploratory_uncalibrated",
            }
        )
    return pd.DataFrame(rows)


def write_decisions(
    effects_csv: Path,
    output_csv: Path,
    thresholds: DecisionThresholds,
    metadata_json: Path | None = None,
    *,
    primary_endpoint: PrimaryEndpoint = "current",
) -> pd.DataFrame:
    """Write a decision table using the declared primary endpoint."""

    decisions = classify_effects(
        pd.read_csv(effects_csv),
        thresholds,
        primary_endpoint=primary_endpoint,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output_csv, index=False)
    if metadata_json is not None:
        metadata_json.write_text(
            json.dumps(
                {
                    "thresholds": asdict(thresholds),
                    "primary_endpoint": primary_endpoint,
                    "decision_rule": "minimum arteries exceeding the primary endpoint threshold",
                    "calcium_role": "exploratory only until independent calibration",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return decisions


__all__ = ["DecisionThresholds", "PrimaryEndpoint", "classify_effects", "write_decisions"]
