from __future__ import annotations

import sys
import types

import pytest

from piconewton_susceptibility.parent import load_parent_api, verified_hydrodynamic_config
from piconewton_susceptibility.source_registry import load_source_registry


class FakeConfig:
    __module__ = "piconewton_v3.types"

    def __init__(self, mode: str = "verified", **kwargs: object):
        self.mode = mode
        self.kwargs = kwargs


def _function(module: str):
    def inner(*args: object, **kwargs: object):
        return None

    inner.__module__ = module
    return inner


def _class(name: str, module: str):
    return type(name, (), {"__module__": module})


def install_fake_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    package = types.ModuleType("piconewton_v3")
    package.__path__ = []  # type: ignore[attr-defined]
    hydro = types.ModuleType("piconewton_v3.hydrodynamics")
    parent_types = types.ModuleType("piconewton_v3.types")
    study_io = types.ModuleType("piconewton_v3.study_io")

    hydro.WomersleySolver = _class("WomersleySolver", hydro.__name__)
    hydro.classical_womersley_solution = _function(hydro.__name__)
    hydro.compute_hydrodynamics = _function(hydro.__name__)
    hydro.isotropic_validation = _function(hydro.__name__)

    parent_types.ArteryCase = _class("ArteryCase", parent_types.__name__)
    parent_types.EndothelialControlVolume = _class(
        "EndothelialControlVolume", parent_types.__name__
    )
    parent_types.FluidProperties = _class("FluidProperties", parent_types.__name__)
    parent_types.HydrodynamicConfig = FakeConfig
    parent_types.V2_ARTERY_CASES = tuple(range(6))
    parent_types.V2_EXPECTED_BLOB_SHA = "9d61c237cda75df338ce0383038f7765c886f503"

    study_io.StudyStore = _class("StudyStore", study_io.__name__)
    study_io.resolve_study_root = _function(study_io.__name__)
    study_io.sha256_file = _function(study_io.__name__)

    for name, module in {
        "piconewton_v3": package,
        "piconewton_v3.hydrodynamics": hydro,
        "piconewton_v3.types": parent_types,
        "piconewton_v3.study_io": study_io,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_adapter_loads_only_frozen_parent_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_parent(monkeypatch)
    api = load_parent_api(load_source_registry())
    assert len(api.V2_ARTERY_CASES) == 6
    assert api.V2_EXPECTED_BLOB_SHA.endswith("886f503")
    config = verified_hydrodynamic_config(api, radial_order=64)
    assert config.mode == "verified"


def test_adapter_rejects_reproduction_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_parent(monkeypatch)
    api = load_parent_api(load_source_registry())
    with pytest.raises(ValueError, match="verified"):
        verified_hydrodynamic_config(api, mode="reproduction")


def test_adapter_rejects_wrong_symbol_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_parent(monkeypatch)
    sys.modules["piconewton_v3.hydrodynamics"].compute_hydrodynamics.__module__ = (
        "piconewton_v3.sensor"
    )
    with pytest.raises(RuntimeError, match="compute_hydrodynamics"):
        load_parent_api(load_source_registry())
