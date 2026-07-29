"""Source-file and Git-blob verification for the successor bootstrap."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .source_registry import SourceRegistry, load_source_registry


def git_blob_sha_bytes(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324 - Git object identity


def git_blob_sha_file(path: Path) -> str:
    return git_blob_sha_bytes(path.read_bytes())


def validate_parent_source(
    repo_root: str | Path,
    registry: SourceRegistry | None = None,
) -> dict[str, Any]:
    registry = registry or load_source_registry()
    root = Path(repo_root).resolve()
    records: list[dict[str, Any]] = []

    expected_files = [registry.data["canonical_parent_artifact"]]
    expected_files.extend(registry.data["verified_parent_interface"]["files"])
    for item in expected_files:
        path = root / item["path"]
        observed = git_blob_sha_file(path) if path.is_file() else None
        expected = item["git_blob_sha"]
        records.append(
            {
                "path": item["path"],
                "expected_git_blob_sha": expected,
                "observed_git_blob_sha": observed,
                "passed": observed == expected,
            }
        )

    passed = all(record["passed"] for record in records)
    return {
        "repository_root": str(root),
        "published_source_commit": registry.data["repository"]["published_source_commit"],
        "registry_sha256": registry.sha256,
        "files": records,
        "passed": passed,
    }
