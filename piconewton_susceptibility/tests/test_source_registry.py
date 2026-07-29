from piconewton_susceptibility.source_registry import load_source_registry


def test_source_registry_is_frozen_and_step2_is_non_scientific() -> None:
    registry = load_source_registry()
    assert registry.expected_v2_blob == "9d61c237cda75df338ce0383038f7765c886f503"
    assert set(registry.allowed_modules) == {
        "piconewton_v3.hydrodynamics",
        "piconewton_v3.types",
        "piconewton_v3.study_io",
    }
    assert registry.data["step2_boundary"]["scientific_calculations_authorized"] is False
    assert len(registry.data["frozen_artery_registry"]) == 6
    assert registry.data["frozen_artery_registry"][0]["native_pressure_gradient_scale_pa_per_m"] == 9000.0
    assert registry.data["frozen_dimensional_constants"]["publication_force_benchmarks_n"] == [
        1e-12,
        1e-11,
    ]
