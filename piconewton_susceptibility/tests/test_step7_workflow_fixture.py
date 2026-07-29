import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from piconewton_v3 import V2_ARTERY_CASES
from piconewton_susceptibility import experiments_workflow as workflow
from piconewton_susceptibility.experiments_core import (
    Step7Config,
    evaluate_susceptibility,
    susceptibility_metrics,
)


@dataclass
class FakeResponses:
    max_residual: float = 1e-15


def _fake_kernels(case, _responses, eta, config):
    frequencies = np.arange(-6, 7)
    kernel = np.zeros((13, 13), dtype=complex)
    factor = (1.0 + V2_ARTERY_CASES.index(case)) * (1.0 + eta)
    for index in range(13):
        if frequencies[index] != 0:
            kernel[index, index] = factor / (1.0 + abs(frequencies[index]))
            kernel[index, 12 - index] = 0.5 * factor
    return frequencies, kernel, config.exact_epsilon**2 * kernel


def _write_step6(root, config):
    rows = []
    for case in V2_ARTERY_CASES:
        frequencies, kernel, _exact = _fake_kernels(case, None, 1e-5 / case.radius_m, config)
        _q, spectrum, waveform = evaluate_susceptibility(
            frequencies, kernel, case.harmonic_coefficients, config.time_points
        )
        rows.append(
            {
                "artery_id": case.artery_id,
                "phi_rms": susceptibility_metrics(waveform, spectrum)["phi_rms"],
            }
        )
    native = root / "native_susceptibility.csv"
    pd.DataFrame(rows).to_csv(native, index=False)
    gate = root / "step6_gate.json"
    gate.write_text(json.dumps({"passed": True}), encoding="utf-8")
    manifest = {
        "status": "complete",
        "allowed_next_step": 7,
        "files": {
            "native_susceptibility.csv": {
                "sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
                "bytes": native.stat().st_size,
            }
        },
    }
    (root / "step6_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_step7_fixture_closes_all_gates(tmp_path, monkeypatch):
    config = Step7Config(
        radial_order=30,
        time_points=128,
        quadrature_nodes=16,
        exact_epsilon=0.08,
    )
    step6 = tmp_path / "step6"
    step6.mkdir()
    _write_step6(step6, config)
    monkeypatch.setattr(workflow, "response_set", lambda _case, _config: FakeResponses())
    monkeypatch.setattr(workflow, "dimensionless_kernels", _fake_kernels)
    result = workflow.run_waveform_experiments(tmp_path / "step7", step6, config)
    assert result["manifest"]["status"] == "complete"
    assert result["manifest"]["allowed_next_step"] == 8
    assert len(result["matrices"]) == 72
    assert len(result["exact"]) == 72
    assert result["controls"].query("family == 'harmonic_removal_rms_matched'").shape[0] == 36
