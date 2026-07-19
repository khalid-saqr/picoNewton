from pathlib import Path
import json
import pytest
from piconewton_v4.calibration import load_parameterization, REQUIRED_SOURCE_GROUPS

def test_incomplete_calibration_is_audited_and_cannot_be_required(tmp_path):
    root=Path(__file__).parents[1]
    source=root/'configs'/'calibration_template.json'
    interface, endpoint, audit=load_parameterization(source)
    assert not audit['complete']
    assert set(audit['missing_source_groups'])==set(REQUIRED_SOURCE_GROUPS)
    assert not endpoint.calibrated
    with pytest.raises(ValueError):
        load_parameterization(source,require_calibrated=True)

def test_complete_calibration_requires_all_sources(tmp_path):
    root=Path(__file__).parents[1]
    raw=json.loads((root/'configs'/'calibration_template.json').read_text())
    raw['calibration_status']='experimentally_calibrated'
    raw['sources']={name:f'independent-source:{name}' for name in REQUIRED_SOURCE_GROUPS}
    path=tmp_path/'calibrated.json'; path.write_text(json.dumps(raw))
    _, endpoint, audit=load_parameterization(path,require_calibrated=True)
    assert audit['complete'] and endpoint.calibrated
