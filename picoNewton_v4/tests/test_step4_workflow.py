from pathlib import Path
import json

import pandas as pd

from piconewton_v4.workflow_step4 import run_step4


ROOT = Path(__file__).resolve().parents[1]


def test_quick_step4_workflow(tmp_path):
    manifest = run_step4(package_root=ROOT, output_root=tmp_path, profile="quick")
    assert manifest["status"] == "passed"
    summary = pd.read_csv(tmp_path / "six_artery_hydrodynamic_summary.csv")
    assert len(summary) == 6
    assert set(summary["artery_id"]) == {
        "aortic_root",
        "thoracic_aorta",
        "femoral",
        "carotid",
        "iliac",
        "brachial",
    }
    validation = json.loads((tmp_path / "step4_validation.json").read_text())
    assert validation["passed"]
    assert validation["max_isotropic_linf_error"] < 1e-7
