import json
from pathlib import Path

import pytest

from piconewton_susceptibility.publication_core import (
    Step10Config,
    validate_step2,
    validate_step9,
)
from step10_fixture import build_step10_fixture


def test_step10_config_validation():
    Step10Config(profile="quick", figure_dpi=180, figure_formats=("png",)).validate()
    with pytest.raises(ValueError):
        Step10Config(figure_dpi=100).validate()


def test_prior_gate_validation_is_fail_closed(tmp_path: Path):
    root = build_step10_fixture(tmp_path / "workflow")
    validate_step2(root / "bootstrap" / "step2")
    validate_step9(root / "step9_robustness_claim_lock")

    gate_path = root / "step9_robustness_claim_lock" / "claim_lock.json"
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    payload["status"] = "revision_required"
    gate_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError):
        validate_step9(root / "step9_robustness_claim_lock")
