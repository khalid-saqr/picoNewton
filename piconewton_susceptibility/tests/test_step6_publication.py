from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from piconewton_susceptibility.susceptibility_core import (
    ExactNativeEvaluator,
    Step6Config,
    second_order_native,
)
from piconewton_susceptibility.susceptibility_workflow import run_susceptibility_inversion
from piconewton_v3 import V2_ARTERY_CASES


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prior_evidence(root: Path, config: Step6Config) -> tuple[Path, Path]:
    step4 = root / "step4"
    step5 = root / "step5"
    step4.mkdir(parents=True)
    step5.mkdir(parents=True)

    validity = pd.DataFrame(
        {
            "artery_id": [case.artery_id for case in V2_ARTERY_CASES],
            "force_valid_epsilon_max_1pct": [0.08] * 6,
        }
    )
    validity_path = step4 / "validity_domains.csv"
    validity.to_csv(validity_path, index=False)
    (step4 / "step4_gate.json").write_text('{"passed":true}', encoding="utf-8")
    (step4 / "step4_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "allowed_next_step": 5,
                "files": {validity_path.name: {"sha256": _digest(validity_path)}},
            }
        ),
        encoding="utf-8",
    )

    arrays: dict[str, np.ndarray] = {}
    for case in V2_ARTERY_CASES:
        second = second_order_native(case, config)
        exact = ExactNativeEvaluator(case, config)
        _, exact_spectrum, exact_waveform = exact.spectrum_and_waveform(0.10)
        arrays[f"{case.artery_id}__second_order__waveform_n"] = second[
            "waveform_n_per_epsilon2"
        ]
        arrays[f"{case.artery_id}__second_order__spectrum"] = second["spectrum"]
        arrays[f"{case.artery_id}__exact_excess__waveform_n"] = exact_waveform
        arrays[f"{case.artery_id}__exact_excess__spectrum"] = exact_spectrum
    archive = step5 / "kernel_archive.npz"
    np.savez_compressed(archive, **arrays)
    (step5 / "step5_gate.json").write_text('{"passed":true}', encoding="utf-8")
    (step5 / "step5_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "allowed_next_step": 6,
                "files": {archive.name: {"sha256": _digest(archive)}},
            }
        ),
        encoding="utf-8",
    )
    return step4, step5


def test_reduced_publication_profile_closes_step6(tmp_path: Path) -> None:
    config = Step6Config(
        profile="publication",
        radial_order=35,
        time_points=256,
        quadrature_nodes=48,
        validation_epsilons=(0.04, 0.08, 0.10),
        inversion_verification_epsilons=(0.04,),
        closure_tolerance=1e-9,
    )
    step4, step5 = _prior_evidence(tmp_path, config)
    result = run_susceptibility_inversion(
        tmp_path / "out", step5, step4, config
    )
    assert result["manifest"]["status"] == "complete"
    assert result["manifest"]["allowed_next_step"] == 7
    assert result["manifest"]["gates"]["passed"] is True
    assert len(result["native"]) == 6
    assert len(result["critical"]) == 24
    assert result["critical"]["full_model_crossing"].isna().all()
