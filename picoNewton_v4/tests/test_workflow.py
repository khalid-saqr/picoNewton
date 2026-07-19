from pathlib import Path
import json
from piconewton_v4.workflow import run_workflow

def test_workflow_structural_gates(tmp_path):
    root=Path(__file__).parents[1]
    manifest=run_workflow(package_root=root,output_root=tmp_path/'out',run_scan=False)
    assert manifest['status']=='passed_structural_validation'
    validation=json.loads((tmp_path/'out'/'validation.json').read_text())
    assert validation['signed_and_exposure_force_classes_separate']
    assert not validation['magnitude_exposure_used_as_signed_load']
    assert not validation['endpoint_calibrated']
    assert not validation['claims_enabled']
