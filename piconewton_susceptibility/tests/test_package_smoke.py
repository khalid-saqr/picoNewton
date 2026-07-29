import piconewton_susceptibility as package


def test_package_import_and_version() -> None:
    assert package.__version__ == "0.1.0"
    assert package.load_source_registry().data["schema_version"] == "1.0.0"
