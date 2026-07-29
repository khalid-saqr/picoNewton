from pathlib import Path

from piconewton_susceptibility.provenance import git_blob_sha_bytes, validate_parent_source
from piconewton_susceptibility.source_registry import SourceRegistry


def test_git_blob_sha_matches_known_git_object() -> None:
    assert git_blob_sha_bytes(b"test content\n") == "d670460b4b4aece5915caf5c68d12f560a9fe3e4"


def test_parent_validation_fails_closed_for_missing_files(tmp_path: Path) -> None:
    data = {
        "repository": {"published_source_commit": "a" * 40},
        "canonical_parent_artifact": {"path": "missing.ipynb", "git_blob_sha": "b" * 40},
        "verified_parent_interface": {"files": []},
    }
    registry = SourceRegistry(data=data, sha256="c" * 64)
    result = validate_parent_source(tmp_path, registry)
    assert result["passed"] is False
    assert result["files"][0]["observed_git_blob_sha"] is None
