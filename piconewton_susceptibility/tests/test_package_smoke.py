from piconewton_susceptibility import (
    Step3Config,
    __version__,
    run_parent_continuity,
    validate_bootstrap_artifacts,
)


def test_package_public_api():
    assert __version__ == "0.3.0"
    assert Step3Config().profile == "publication"
    assert callable(run_parent_continuity)
    assert callable(validate_bootstrap_artifacts)
