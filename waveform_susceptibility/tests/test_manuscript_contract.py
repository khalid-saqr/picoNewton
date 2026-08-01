import numpy as np

from piconewton_waveform_susceptibility import AnalysisConfig, run_analysis
from piconewton_waveform_susceptibility.figures import _pair_statistics


def _separation_residual_percent(crossed):
    matrix = (
        crossed[crossed["condition"] == "physiological"]
        .pivot(index="vessel_id", columns="waveform_id", values="phi_rms")
        .to_numpy()
    )
    logs = np.log(matrix)
    artery_factor = np.exp(logs.mean(axis=1, keepdims=True))
    waveform_factor = np.exp(logs.mean(axis=0, keepdims=True) - logs.mean())
    return 100.0 * (matrix / (artery_factor * waveform_factor) - 1.0)


def test_default_analysis_reproduces_manuscript_results(tmp_path):
    config = AnalysisConfig()
    assert config.radial_order == 150
    assert config.time_points == 2048
    assert config.quadrature_nodes == 256
    assert config.validation_epsilon == 0.08
    assert config.harmonics == 6

    result = run_analysis(tmp_path, config)
    atlas = result["atlas"].set_index("artery_id")
    expected_susceptibility = {
        "brachial": 9.36e-7,
        "carotid": 1.34e-7,
        "femoral": 6.98e-8,
        "iliac": 4.14e-8,
        "thoracic_aorta": 1.21e-9,
        "aortic_root": 5.54e-10,
    }
    expected_force_pn = {
        "aortic_root": 0.112,
        "thoracic_aorta": 0.061,
        "carotid": 0.042,
        "femoral": 0.032,
        "iliac": 0.025,
        "brachial": 0.012,
    }
    for artery, expected in expected_susceptibility.items():
        assert np.isclose(atlas.loc[artery, "phi_rms"], expected, rtol=0.005)
    for artery, expected in expected_force_pn.items():
        assert np.isclose(
            atlas.loc[artery, "predicted_rms_at_epsilon_0p08_pn"],
            expected,
            atol=0.0005,
        )

    physiological = result["crossed"][result["crossed"]["condition"] == "physiological"]
    assert len(physiological) == 36
    residual = _separation_residual_percent(result["crossed"])
    assert np.isclose(np.max(np.abs(residual)), 0.8833, atol=0.0001)

    matched = result["controls"][
        result["controls"]["family"] == "harmonic_removal_rms_matched"
    ].copy()
    matched["harmonic"] = matched["control"].str.extract(r"h([1-6])").astype(int)
    fundamental = 100.0 * matched[matched["harmonic"] == 1]["fractional_change"]
    assert np.isclose(fundamental.min(), -49.0338, atol=0.001)
    assert np.isclose(fundamental.max(), -41.3674, atol=0.001)

    _spectrum, _dominant, leading_pairs = _pair_statistics(result["pairs"])
    expected_leading = {
        0: (-1, 1, 80.0),
        1: (-1, 2, 85.0),
        2: (1, 1, 84.0),
        3: (1, 2, 94.0),
        6: (1, 5, 95.0),
        9: (4, 5, 85.0),
        10: (4, 6, 107.0),
        11: (5, 6, 100.0),
        12: (6, 6, 100.0),
    }
    for output, expected in expected_leading.items():
        m, n, projection = leading_pairs[output][0]
        assert (m, n) == expected[:2]
        assert np.isclose(projection, expected[2], atol=0.6)

    summary = result["summary"]
    law = summary["reduced_law"]
    assert summary["crossed_entries"] == 36
    assert summary["operator_samples"] == 12
    assert summary["held_out_predictions"] == 1068
    assert summary["constitutive_paths"] == 9
    assert np.isclose(law["retained_energy"], 0.9999860359, atol=1e-10)
    assert np.isclose(law["median_relative_error"], 0.02231464, atol=1e-8)
    assert np.isclose(law["p90_relative_error"], 0.10188632, atol=1e-8)
    assert np.isclose(law["maximum_relative_error"], 0.16296206, atol=1e-8)

    robustness = result["robustness"]
    assert set(robustness["constitutive_path"]) == {
        "reciprocal",
        "beta_low",
        "gamma_low",
        "gamma_only",
        "beta_high_gamma_low",
        "beta_low_gamma_high",
        "delta_low",
        "delta_high",
        "beta_only",
    }
    beta_only = robustness[robustness["constitutive_path"] == "beta_only"]
    assert beta_only["null_control"].all()
    assert beta_only["relative_amplitude_to_reciprocal"].max() < 1e-12
