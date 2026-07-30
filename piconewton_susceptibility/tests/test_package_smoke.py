from piconewton_susceptibility import (
    __version__,
    validate_bootstrap_artifacts,
)


def test_package_public_api():
    assert __version__ == "0.9.0"
    assert callable(validate_bootstrap_artifacts)
