from pathlib import Path

from piconewton_v4.runtime import create_runtime


def test_unique_runtime_directories(tmp_path: Path):
    drive = tmp_path / "drive"
    local = tmp_path / "local"
    first = create_runtime(drive_root=drive, local_base=local)
    second = create_runtime(drive_root=drive, local_base=local)
    assert first.run_id != second.run_id
    assert first.persistent_root.exists()
    assert second.persistent_root.exists()
    assert (first.manifests / "runtime_manifest.json").exists()
