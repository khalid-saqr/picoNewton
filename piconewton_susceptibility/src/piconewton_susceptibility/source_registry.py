"""Load and validate the frozen parent-source registry."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceRegistry:
    """Validated immutable source-chain metadata."""

    data: dict[str, Any]
    sha256: str

    @property
    def expected_v2_blob(self) -> str:
        return str(self.data["canonical_parent_artifact"]["git_blob_sha"])

    @property
    def allowed_modules(self) -> tuple[str, ...]:
        return tuple(self.data["verified_parent_interface"]["allowed_modules"])

    @property
    def forbidden_module_prefixes(self) -> tuple[str, ...]:
        return tuple(self.data["verified_parent_interface"]["forbidden_module_prefixes"])


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _validate_sha(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError(f"{name} must be a 40-character Git SHA")
    int(value, 16)


def validate_source_registry(data: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "publication",
        "repository",
        "canonical_parent_artifact",
        "verified_parent_interface",
        "frozen_dimensional_constants",
        "frozen_artery_registry",
        "step2_boundary",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"source registry missing keys: {sorted(missing)}")

    publication = data["publication"]
    if publication.get("doi") != "10.1038/s41598-026-47474-x":
        raise ValueError("unexpected parent DOI")

    repository = data["repository"]
    if repository.get("full_name") != "khalid-saqr/picoNewton":
        raise ValueError("unexpected parent repository")
    _validate_sha(repository.get("published_source_commit"), "published_source_commit")

    artifact = data["canonical_parent_artifact"]
    if artifact.get("path") != "picoNewton_v2.ipynb":
        raise ValueError("unexpected canonical parent artifact")
    _validate_sha(artifact.get("git_blob_sha"), "canonical parent blob")

    interface = data["verified_parent_interface"]
    if interface.get("solver_mode") != "verified":
        raise ValueError("successor parent interface must be verified mode")
    allowed = set(interface.get("allowed_modules", []))
    expected_allowed = {
        "piconewton_v3.hydrodynamics",
        "piconewton_v3.types",
        "piconewton_v3.study_io",
    }
    if allowed != expected_allowed:
        raise ValueError("allowed parent-module set has drifted")
    for record in interface.get("files", []):
        _validate_sha(record.get("git_blob_sha"), f"blob for {record.get('path')}")

    arteries = data["frozen_artery_registry"]
    if len(arteries) != 6 or len({item["artery_id"] for item in arteries}) != 6:
        raise ValueError("frozen artery registry must contain six unique arteries")
    if any(len(item["harmonic_coefficients"]) != 6 for item in arteries):
        raise ValueError("every frozen artery must contain six harmonics")

    constants = data["frozen_dimensional_constants"]
    if constants.get("publication_force_benchmarks_n") != [1e-12, 1e-11]:
        raise ValueError("publication force benchmarks have drifted")
    if data["step2_boundary"].get("scientific_calculations_authorized") is not False:
        raise ValueError("Step 2 must not authorize scientific calculations")


def load_source_registry(path: str | Path | None = None) -> SourceRegistry:
    if path is None:
        resource = files("piconewton_susceptibility").joinpath("data", "source_registry.json")
        raw = resource.read_bytes()
    else:
        raw = Path(path).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    validate_source_registry(data)
    digest = hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()
    return SourceRegistry(data=data, sha256=digest)
