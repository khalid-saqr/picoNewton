from pathlib import Path

from piconewton_v4.sources import validate_cellml_sources


def test_committed_cellml_sources():
    package_root = Path(__file__).resolve().parents[1]
    report = validate_cellml_sources(package_root)
    assert report["ok"], report
    assert report["license"] == "Creative Commons Attribution 3.0 Unported"
    assert len(report["files"]) == 3
