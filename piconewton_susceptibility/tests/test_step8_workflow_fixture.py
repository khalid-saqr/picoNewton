import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES

from piconewton_susceptibility.reduction_core import Step8Config
from piconewton_susceptibility.reduction_workflow import run_reduction_study


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_step7(root: Path) -> None:
    rows = []
    arrays = {}
    frequencies = np.concatenate((np.arange(-6, 0), np.arange(1, 7)))
    left = (1.0 + np.abs(frequencies)) ** -0.5 * np.exp(0.07j * frequencies)
    right = (1.0 + np.abs(frequencies)) ** -0.5 * np.exp(-0.03j * frequencies)
    base = np.outer(left, right)
    for matrix_type in ("hydrodynamic", "physiological"):
        for vessel_index, case in enumerate(V2_ARTERY_CASES):
            alpha = 22.0 / (1.0 + 0.45 * vessel_index)
            eta = 0.002361111 if matrix_type == "hydrodynamic" else 0.0007 * (1.4**vessel_index)
            scale = 0.4 * alpha**-2.0 * eta**1.95
            kernel = scale * base
            arrays[f"{matrix_type}__{case.artery_id}__kernel2"] = kernel
            for waveform in V2_ARTERY_CASES:
                rows.append(
                    {
                        "matrix_type": matrix_type,
                        "vessel_id": case.artery_id,
                        "vessel_name": case.name,
                        "waveform_id": waveform.artery_id,
                        "waveform_name": waveform.name,
                        "native_diagonal": case.artery_id == waveform.artery_id,
                        "alpha": alpha,
                        "eta": eta,
                        "phi_rms": 1.0,
                    }
                )
    crossed = root / "crossed_susceptibility.csv"
    pd.DataFrame(rows).to_csv(crossed, index=False)
    archive = root / "step7_archive.npz"
    np.savez_compressed(archive, **arrays)
    gate = root / "step7_gate.json"
    gate.write_text(json.dumps({"passed": True}), encoding="utf-8")
    manifest = {
        "status": "complete",
        "allowed_next_step": 8,
        "files": {
            crossed.name: {"sha256": _sha(crossed), "bytes": crossed.stat().st_size},
            archive.name: {"sha256": _sha(archive), "bytes": archive.stat().st_size},
            gate.name: {"sha256": _sha(gate), "bytes": gate.stat().st_size},
        },
    }
    (root / "step7_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_step8_fixture_selects_rank_one(tmp_path):
    step7 = tmp_path / "step7"
    step7.mkdir()
    _write_step7(step7)
    result = run_reduction_study(
        tmp_path / "step8",
        step7,
        Step8Config(profile="quick", phase_scrambles=4),
    )
    assert result["manifest"]["status"] == "complete"
    assert result["manifest"]["allowed_next_step"] == 9
    assert result["law"]["selected_rank"] == 1
    assert result["manifest"]["gates"]["rank_one_selected"]
