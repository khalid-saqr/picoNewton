import hashlib
import json
from pathlib import Path

import numpy as np
from piconewton_v3 import V2_ARTERY_CASES

from piconewton_susceptibility.reduction_core import (
    fit_power_law,
    kernel_scale,
    truncated_kernel,
    universal_kernel,
)
from piconewton_susceptibility.robustness_core import (
    Step9Config,
    alpha_for_case,
    derive_general_hierarchy,
    hierarchy_kernel,
    native_eta,
)
from piconewton_susceptibility.robustness_workflow import run_robustness_study


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _step8_fixture(root: Path) -> None:
    records = []
    for case in V2_ARTERY_CASES:
        basis = derive_general_hierarchy(case, 60, 1.0)
        for eta in (2.361111e-3, native_eta(case)):
            kernel = hierarchy_kernel(basis, eta, 96, 1.0, 1.0)
            records.append((alpha_for_case(case), eta, kernel_scale(kernel), kernel))
    parameters = fit_power_law(
        np.asarray([row[0] for row in records]),
        np.asarray([row[1] for row in records]),
        np.asarray([row[2] for row in records]),
    )
    universal = universal_kernel(row[3] for row in records)
    selected, singular_values, _ = truncated_kernel(universal, 1)
    np.savez_compressed(
        root / "step8_reduced_law.npz",
        universal_kernel=universal,
        selected_kernel=selected,
        singular_values=singular_values,
        scale_parameters=parameters,
    )
    law = {
        "selected_rank": 1,
        "prefactor": float(np.exp(parameters[0])),
        "alpha_exponent": float(parameters[1]),
        "eta_exponent": float(parameters[2]),
    }
    (root / "reduced_law.json").write_text(json.dumps(law), encoding="utf-8")
    (root / "step8_gate.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    names = ("step8_reduced_law.npz", "reduced_law.json")
    files = {
        name: {
            "sha256": _sha256(root / name),
            "bytes": (root / name).stat().st_size,
        }
        for name in names
    }
    (root / "step8_manifest.json").write_text(
        json.dumps(
            {"status": "complete", "allowed_next_step": 9, "files": files}
        ),
        encoding="utf-8",
    )


def test_quick_workflow_closes_and_locks_claim(tmp_path):
    step8 = tmp_path / "step8"
    output = tmp_path / "step9"
    step8.mkdir()
    _step8_fixture(step8)
    config = Step9Config(
        profile="quick",
        radial_order=60,
        quadrature_nodes=96,
        resolution_pairs=((50, 72), (70, 112)),
    )
    result = run_robustness_study(output, step8, config)
    assert result["manifest"]["status"] == "complete"
    assert result["manifest"]["allowed_next_step"] == 10
    assert result["gates"]["amplitude_claim_restricted_to_reciprocal"]
    assert result["gates"]["beta_only_exact_null_passed"]
    assert "separate constitutive amplitude factor" in result["claim_lock"][
        "required_qualifier"
    ]
    assert (output / "step9_archive.npz").is_file()
    assert (output / "step9_manifest.json").is_file()
