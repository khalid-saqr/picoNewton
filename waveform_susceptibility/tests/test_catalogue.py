from piconewton_waveform_susceptibility import waveform_catalogue


def test_waveform_catalogue_has_declared_coverage():
    catalogue = waveform_catalogue()
    assert len(catalogue) == 89
    families = {item["family"] for item in catalogue}
    assert families == {
        "native",
        "single_tone",
        "two_tone",
        "sparse_three_tone",
        "spectral_slope",
        "phase_challenge",
    }
