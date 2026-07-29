from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from piconewton_susceptibility.bootstrap import BootstrapConfig, bootstrap_environment
from piconewton_susceptibility.validation import validate_bootstrap_artifacts


class FakeStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def initialize_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, relative: str, data: object) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def write_checksums(self, relative_root: str, output_relative: str) -> Path:
        root = self.root / relative_root
        output = root / output_relative
        records = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path == output:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append(f"{digest}  {path.relative_to(root).as_posix()}")
        output.write_text("\n".join(records) + "\n", encoding="utf-8")
        return output


def _fake_api() -> SimpleNamespace:
    return SimpleNamespace(
        StudyStore=FakeStore,
        resolve_study_root=lambda **kwargs: (Path(kwargs["local_root"]), "local"),
    )


def test_development_bootstrap_is_explicitly_non_claim_bearing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "piconewton_susceptibility.bootstrap.load_parent_api", lambda registry: _fake_api()
    )
    result = bootstrap_environment(
        BootstrapConfig(
            repo_root=tmp_path,
            storage_mode="local",
            local_root=tmp_path / "outputs",
            development_skip_parent_validation=True,
        )
    )
    manifest = result["manifest"]
    gate = result["completion_gate"]
    assert manifest["claim_bearing"] is False
    assert manifest["scientific_calculations_run"] is False
    assert manifest["scientific_calculations_authorized"] is False
    assert gate["passed"] is False
    assert gate["allowed_next_step"] is None
    assert result["validation"]["passed"] is True


def test_claim_bearing_bootstrap_requires_final_integrity_gate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "piconewton_susceptibility.bootstrap.load_parent_api", lambda registry: _fake_api()
    )
    monkeypatch.setattr(
        "piconewton_susceptibility.bootstrap.validate_parent_source",
        lambda repo_root, registry: {
            "repository_root": str(repo_root),
            "registry_sha256": registry.sha256,
            "passed": True,
            "files": [],
        },
    )
    result = bootstrap_environment(
        BootstrapConfig(
            repo_root=tmp_path,
            storage_mode="local",
            local_root=tmp_path / "outputs",
        )
    )
    assert result["manifest"]["claim_bearing"] is True
    assert result["completion_gate"]["passed"] is True
    assert result["completion_gate"]["allowed_next_step"] == 3
    assert result["validation"]["passed"] is True

    runtime_path = Path(result["runtime_validation_path"])
    runtime_path.write_text("{}", encoding="utf-8")
    validation = validate_bootstrap_artifacts(
        result["bootstrap_root"], expected_storage_mode="local"
    )
    assert validation["passed"] is False
    assert validation["checksum_validation"]["passed"] is False
