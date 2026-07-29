from pathlib import Path

from piconewton_susceptibility.validation import storage_round_trip_probe


def test_storage_round_trip_probe_removes_probe_files(tmp_path: Path) -> None:
    record = storage_round_trip_probe(tmp_path)
    assert record["passed"] is True
    assert record["content_match"] is True
    assert record["delete_confirmed"] is True
    assert not list((tmp_path / "bootstrap" / "step2").glob(".storage_probe_*"))
