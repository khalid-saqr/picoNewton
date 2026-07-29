from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from piconewton_susceptibility.bootstrap import BootstrapConfig, bootstrap_environment


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


def test_development_bootstrap_is_explicitly_non_claim_bearing(
    tmp_path: Path, monkeypatch
) -> None:
    fake_api = SimpleNamespace(
        StudyStore=FakeStore,
        resolve_study_root=lambda **kwargs: (Path(kwargs["local_root"]), "local"),
    )
    monkeypatch.setattr(
        "piconewton_susceptibility.bootstrap.load_parent_api", lambda registry: fake_api
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
    assert manifest["claim_bearing"] is False
    assert manifest["scientific_calculations_run"] is False
    assert manifest["scientific_calculations_authorized"] is False
    assert manifest["allowed_next_step"] is None
    assert Path(result["manifest_path"]).is_file()
    assert Path(result["checksums_path"]).is_file()
