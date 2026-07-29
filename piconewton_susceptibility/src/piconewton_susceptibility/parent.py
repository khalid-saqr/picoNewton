"""Fail-closed adapter to the verified parent hydrodynamic interface."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from .source_registry import SourceRegistry, load_source_registry


@dataclass(frozen=True)
class ParentAPI:
    WomersleySolver: type
    classical_womersley_solution: Any
    compute_hydrodynamics: Any
    isotropic_validation: Any
    ArteryCase: type
    EndothelialControlVolume: type
    FluidProperties: type
    HydrodynamicConfig: type
    V2_ARTERY_CASES: tuple[Any, ...]
    V2_EXPECTED_BLOB_SHA: str
    StudyStore: type
    resolve_study_root: Any
    sha256_file: Any


def _assert_origin(symbol: Any, allowed_module: str, name: str) -> None:
    origin = getattr(symbol, "__module__", allowed_module)
    if origin != allowed_module:
        raise RuntimeError(f"{name} resolved from {origin}, expected {allowed_module}")


def load_parent_api(registry: SourceRegistry | None = None) -> ParentAPI:
    """Import only the hydrodynamic and generic-storage surfaces approved in Step 1.

    Importing a Python submodule executes its package initializer. The parent initializer
    may incidentally import later modules, so the enforceable boundary is the origin of
    every callable and type exposed by this adapter, not the global ``sys.modules`` set.
    """
    registry = registry or load_source_registry()

    hydro = importlib.import_module("piconewton_v3.hydrodynamics")
    types = importlib.import_module("piconewton_v3.types")
    study_io = importlib.import_module("piconewton_v3.study_io")

    selected = {
        "WomersleySolver": (hydro.WomersleySolver, "piconewton_v3.hydrodynamics"),
        "classical_womersley_solution": (
            hydro.classical_womersley_solution,
            "piconewton_v3.hydrodynamics",
        ),
        "compute_hydrodynamics": (hydro.compute_hydrodynamics, "piconewton_v3.hydrodynamics"),
        "isotropic_validation": (hydro.isotropic_validation, "piconewton_v3.hydrodynamics"),
        "ArteryCase": (types.ArteryCase, "piconewton_v3.types"),
        "EndothelialControlVolume": (types.EndothelialControlVolume, "piconewton_v3.types"),
        "FluidProperties": (types.FluidProperties, "piconewton_v3.types"),
        "HydrodynamicConfig": (types.HydrodynamicConfig, "piconewton_v3.types"),
        "StudyStore": (study_io.StudyStore, "piconewton_v3.study_io"),
        "resolve_study_root": (study_io.resolve_study_root, "piconewton_v3.study_io"),
        "sha256_file": (study_io.sha256_file, "piconewton_v3.study_io"),
    }
    for name, (symbol, allowed_module) in selected.items():
        _assert_origin(symbol, allowed_module, name)

    if getattr(types, "V2_EXPECTED_BLOB_SHA") != registry.expected_v2_blob:
        raise RuntimeError("installed parent package has an unexpected v2 blob guard")

    api = ParentAPI(
        WomersleySolver=hydro.WomersleySolver,
        classical_womersley_solution=hydro.classical_womersley_solution,
        compute_hydrodynamics=hydro.compute_hydrodynamics,
        isotropic_validation=hydro.isotropic_validation,
        ArteryCase=types.ArteryCase,
        EndothelialControlVolume=types.EndothelialControlVolume,
        FluidProperties=types.FluidProperties,
        HydrodynamicConfig=types.HydrodynamicConfig,
        V2_ARTERY_CASES=tuple(types.V2_ARTERY_CASES),
        V2_EXPECTED_BLOB_SHA=types.V2_EXPECTED_BLOB_SHA,
        StudyStore=study_io.StudyStore,
        resolve_study_root=study_io.resolve_study_root,
        sha256_file=study_io.sha256_file,
    )
    if len(api.V2_ARTERY_CASES) != 6:
        raise RuntimeError("parent hydrodynamic registry must contain exactly six arteries")
    return api


def verified_hydrodynamic_config(api: ParentAPI, **overrides: Any) -> Any:
    """Create a parent configuration while prohibiting reproduction mode for new results."""
    if overrides.get("mode", "verified") != "verified":
        raise ValueError("successor calculations require parent solver mode='verified'")
    clean = {key: value for key, value in overrides.items() if key != "mode"}
    return api.HydrodynamicConfig(mode="verified", **clean)
