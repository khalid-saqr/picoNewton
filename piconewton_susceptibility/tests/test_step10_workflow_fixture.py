import hashlib
import json
from pathlib import Path

from piconewton_susceptibility.publication_core import Step10Config
from piconewton_susceptibility.publication_workflow import run_publication_archive
from step10_fixture import build_step10_fixture


def test_step10_cold_fixture_archive(tmp_path: Path):
    workflow = build_step10_fixture(tmp_path / "workflow")
    output = tmp_path / "publication"
    result = run_publication_archive(
        output,
        workflow,
        config=Step10Config(
            profile="quick",
            figure_dpi=180,
            figure_formats=("png",),
        ),
    )
    manifest = result["manifest"]
    assert manifest["workflow_complete"] is True
    assert manifest["gates"]["passed"] is True
    assert len(list((output / "figures" / "main").glob("*.png"))) == 6
    assert len(list((output / "figures" / "supplementary").glob("*.png"))) == 3
    assert len(list((output / "figures" / "source").glob("*.csv"))) >= 18
    assert len(list((output / "tables" / "main").glob("*.csv"))) >= 5
    assert len(list((output / "tables" / "supplementary").glob("*.csv"))) >= 6

    archive = output / "publication_archive.zip"
    expected = (output / "publication_archive.sha256").read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert expected == actual == manifest["archive_sha256"]

    claim = json.loads((output / "locked_science" / "claim_lock.json").read_text())
    assert claim["status"] == "locked"
