from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from piconewton_v3.study_io import StudyStore, deterministic_run_id, safe_relative_path


def test_deterministic_run_id() -> None:
    a = deterministic_run_id({"profile": "quick"}, "abc")
    b = deterministic_run_id({"profile": "quick"}, "abc")
    c = deterministic_run_id({"profile": "publication"}, "abc")
    assert a == b
    assert a != c


def test_path_traversal_is_rejected() -> None:
    for invalid in ("../secret", "/absolute", "a/../../b"):
        with pytest.raises(ValueError):
            safe_relative_path(invalid)


def test_store_manifest_resume_and_checksums(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study")
    store.initialize_layout()
    config = {"profile": "test"}
    run_id, root = store.create_run(config, "abc", "v2sha", "verified", 1)
    same_id, same_root = store.create_run(config, "abc", "v2sha", "verified", 1)
    assert run_id == same_id
    assert root == same_root

    store.write_csv(
        f"runs/{run_id}/summaries/table.csv", pd.DataFrame({"x": [1, 2]})
    )
    store.write_npz(
        f"runs/{run_id}/fields/signals.npz", signal=np.arange(4)
    )
    store.register_file(run_id, "summaries/table.csv", "output")
    store.register_file(run_id, "fields/signals.npz", "output")
    store.set_status(run_id, "complete")
    checksum = store.write_checksums(
        f"runs/{run_id}", "provenance/checksums.sha256"
    )
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["output_files"]) == 2
    assert checksum.is_file()
