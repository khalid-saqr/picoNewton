"""Step 5 source-model verification for the Piezo1 Markov chain.

This workflow verifies the uncoupled Ogiermann et al. four-state channel model.
It does not accept hydrodynamic, membrane, force, or WSS inputs.
"""
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.integrate import solve_ivp

from .piezo1 import (
    Piezo1Parameters,
    ProtocolSegment,
    equilibrated_state,
    generator_matrix,
    normalized_current_surrogate,
    propagate_constant,
    simulate_protocol,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_committed_cellml(package_root: Path) -> dict[str, object]:
    """Check the vendored source identity and exact parameter transcription."""
    source_root = package_root / "external" / "cellml" / "ogiermann_2025"
    channel = source_root / "models" / "channels" / "ogiermann_2025_piezo1.cellml"
    params_file = source_root / "models" / "channels" / "ogiermann_2025_piezo1_params.cellml"
    units = source_root / "units.cellml"
    source_metadata = json.loads((source_root / "SOURCE.json").read_text(encoding="utf-8"))

    namespace = {"c": "http://www.cellml.org/cellml/1.1#"}
    root = ET.parse(params_file).getroot()
    observed: dict[str, float] = {}
    for variable in root.findall(".//c:variable", namespace):
        value = variable.attrib.get("initial_value")
        if value is not None:
            observed[variable.attrib["name"]] = float(value)

    expected_params = Piezo1Parameters()
    expected = {
        name: float(getattr(expected_params, name))
        for name in (
            "r1", "r3", "r4", "r5", "r6", "r7", "r8",
            "ce1", "ce3", "ce4", "cm4", "ce5", "ce6", "ce7", "ce8",
        )
    }
    differences = {name: abs(observed[name] - value) for name, value in expected.items()}
    files = [channel, params_file, units]
    channel_text = channel.read_text(encoding="utf-8")
    required_tokens = [
        "P_Open", "P_Closed", "P_I1", "P_I2", "pressure", "gradmu",
        "p_O_I1", "p_I1_O", "p_I1_C", "p_C_I1", "p_C_O",
        "p_O_C", "p_I2_O", "p_O_I2",
    ]
    missing_tokens = [token for token in required_tokens if token not in channel_text]
    return {
        "workspace_revision": source_metadata["workspace_revision"],
        "channel_revision": source_metadata["channel_source_revision"],
        "license": source_metadata["license"],
        "files": [
            {
                "path": path.relative_to(package_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "maximum_parameter_abs_difference": max(differences.values()),
        "missing_required_equation_tokens": missing_tokens,
        "passed": bool(max(differences.values()) == 0.0 and not missing_tokens),
    }


def figure3_voltage_clamp() -> dict[str, object]:
    """Reproduce the source paper's 500-ms saturating-pressure voltage protocol."""
    voltages = np.array([-80.0, -60.0, -40.0, -20.0, 20.0, 40.0, 50.0])
    rows = []
    traces = {}
    for voltage in voltages:
        initial = equilibrated_state(voltage)
        result = simulate_protocol(
            [ProtocolSegment(500.0, 70.0, voltage, "pressure")],
            initial_state=initial,
            dt_ms=1.0,
        )
        p_open = result["P_Open"]
        current = normalized_current_surrogate(p_open, voltage)
        peak_current = float(np.max(np.abs(current)))
        peak_open = float(np.max(p_open))
        end_open = float(p_open[-1])
        rows.append(
            {
                "voltage_mv": float(voltage),
                "peak_open_probability": peak_open,
                "end_open_probability": end_open,
                "inactivation_fraction": float((peak_open - end_open) / max(peak_open, 1e-30)),
                "peak_abs_current_surrogate": peak_current,
                "end_to_peak_current_ratio": float(np.abs(current[-1]) / max(peak_current, 1e-30)),
            }
        )
        traces[str(int(voltage))] = {
            "time_ms": result["time_ms"].tolist(),
            "P_Open": p_open.tolist(),
            "normalized_abs_current": (np.abs(current) / max(peak_current, 1e-30)).tolist(),
        }

    negative = np.asarray([row["inactivation_fraction"] for row in rows if row["voltage_mv"] < 0])
    positive = np.asarray([row["inactivation_fraction"] for row in rows if row["voltage_mv"] > 0])
    validation = {
        "negative_voltage_mean_inactivation": float(np.mean(negative)),
        "positive_voltage_mean_inactivation": float(np.mean(positive)),
        "positive_voltage_reduces_inactivation": bool(np.mean(positive) < np.mean(negative)),
        "all_probabilities_bounded": bool(
            all(0.0 <= row["peak_open_probability"] <= 1.0 for row in rows)
        ),
    }
    return {"rows": rows, "traces": traces, "validation": validation}


def repeated_pressure_diagnostic() -> dict[str, object]:
    """Qualitative repeated-pressure diagnostic.

    The article specifies a 1-s cadence, but the channel-only CellML files do
    not encode the original pulse width. The 300-ms width is therefore a
    declared diagnostic setting, not an exact Figure-4 reproduction gate.
    """
    def pulse_train(voltage_mv: float, pulses: int = 6) -> list[ProtocolSegment]:
        segments = [ProtocolSegment(20_000.0, 0.0, voltage_mv, "equilibrate")]
        for index in range(pulses):
            segments.append(ProtocolSegment(300.0, 70.0, voltage_mv, f"pulse_{index + 1}"))
            segments.append(ProtocolSegment(700.0, 0.0, voltage_mv, f"rest_{index + 1}"))
        return segments

    def peaks(result: dict[str, np.ndarray], prefix: str) -> np.ndarray:
        values = []
        index = 1
        while np.any(result["label"] == f"{prefix}{index}"):
            mask = result["label"] == f"{prefix}{index}"
            values.append(float(np.max(result["P_Open"][mask])))
            index += 1
        return np.asarray(values)

    negative = simulate_protocol(pulse_train(-60.0), dt_ms=1.0)
    positive = simulate_protocol(pulse_train(60.0), dt_ms=1.0)
    negative_peaks = peaks(negative, "pulse_")
    positive_peaks = peaks(positive, "pulse_")

    alternating_segments = [ProtocolSegment(20_000.0, 0.0, -60.0, "equilibrate")]
    for index in range(6):
        alternating_segments.extend(
            [
                ProtocolSegment(300.0, 70.0, -60.0, f"negative_pulse_{index + 1}"),
                ProtocolSegment(200.0, 0.0, -60.0, "negative_rest"),
                ProtocolSegment(300.0, 70.0, 60.0, f"positive_pulse_{index + 1}"),
                ProtocolSegment(200.0, 0.0, -60.0, "positive_rest"),
            ]
        )
    alternating = simulate_protocol(alternating_segments, dt_ms=1.0)
    alternating_peaks = peaks(alternating, "negative_pulse_")

    validation = {
        "pulse_width_ms": 300.0,
        "pulse_period_ms": 1000.0,
        "negative_last_to_first": float(negative_peaks[-1] / negative_peaks[0]),
        "positive_last_to_first": float(positive_peaks[-1] / positive_peaks[0]),
        "alternating_negative_last_to_first": float(alternating_peaks[-1] / alternating_peaks[0]),
        "negative_desensitizes": bool(negative_peaks[-1] / negative_peaks[0] < 0.75),
        "positive_preserves_more_than_negative": bool(
            positive_peaks[-1] / positive_peaks[0] > negative_peaks[-1] / negative_peaks[0] + 0.15
        ),
        "alternating_preserves_more_than_negative": bool(
            alternating_peaks[-1] / alternating_peaks[0] > negative_peaks[-1] / negative_peaks[0] + 0.15
        ),
        "exact_figure4_reproduction_claimed": False,
    }
    return {
        "negative_peaks": negative_peaks.tolist(),
        "positive_peaks": positive_peaks.tolist(),
        "alternating_negative_peaks": alternating_peaks.tolist(),
        "validation": validation,
    }


def numerical_validation() -> dict[str, object]:
    pressures = [0.0, 25.0, 70.0]
    voltages = [-80.0, -60.0, -40.0, 0.0, 40.0, 60.0]
    max_column_sum = 0.0
    max_semigroup_error = 0.0
    max_expm_vs_ivp_error = 0.0
    min_probability = 1.0
    max_probability_sum_error = 0.0

    for pressure in pressures:
        for voltage in voltages:
            matrix = generator_matrix(pressure, voltage)
            max_column_sum = max(max_column_sum, float(np.max(np.abs(np.sum(matrix, axis=0)))))
            initial = equilibrated_state(voltage)
            whole = propagate_constant(initial, 1000.0, pressure, voltage)
            split = propagate_constant(initial, 400.0, pressure, voltage)
            split = propagate_constant(split, 600.0, pressure, voltage)
            max_semigroup_error = max(max_semigroup_error, float(np.max(np.abs(whole - split))))

            ivp = solve_ivp(
                lambda _time, state: matrix @ state,
                (0.0, 1000.0),
                initial,
                method="DOP853",
                rtol=1e-11,
                atol=1e-13,
                t_eval=[1000.0],
            )
            if not ivp.success:
                raise RuntimeError(ivp.message)
            max_expm_vs_ivp_error = max(
                max_expm_vs_ivp_error, float(np.max(np.abs(whole - ivp.y[:, -1])))
            )

            sampled = simulate_protocol(
                [ProtocolSegment(1000.0, pressure, voltage)],
                initial_state=initial,
                dt_ms=2.0,
            )["states"]
            min_probability = min(min_probability, float(np.min(sampled)))
            max_probability_sum_error = max(
                max_probability_sum_error,
                float(np.max(np.abs(np.sum(sampled, axis=1) - 1.0))),
            )

    passed = bool(
        max_column_sum < 1e-14
        and max_semigroup_error < 1e-10
        and max_expm_vs_ivp_error < 1e-9
        and min_probability >= -1e-12
        and max_probability_sum_error < 1e-10
    )
    return {
        "max_generator_column_sum": max_column_sum,
        "max_semigroup_error": max_semigroup_error,
        "max_matrix_exponential_vs_dop853_error": max_expm_vs_ivp_error,
        "min_probability": min_probability,
        "max_probability_sum_error": max_probability_sum_error,
        "passed": passed,
    }


def run_step5(*, package_root: Path, output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=False)
    source = validate_committed_cellml(package_root)
    figure3 = figure3_voltage_clamp()
    repeated = repeated_pressure_diagnostic()
    numerical = numerical_validation()

    passed = all(
        bool(value)
        for value in (
            source["passed"],
            numerical["passed"],
            figure3["validation"]["positive_voltage_reduces_inactivation"],
            figure3["validation"]["all_probabilities_bounded"],
        )
    )
    payload = {
        "step": 5,
        "status": "passed" if passed else "failed",
        "source_model": "Ogiermann et al. four-state Piezo1 Markov model",
        "source_doi": "10.1113/JP288666",
        "scope": "uncoupled channel source-model verification only",
        "coupling_executed": False,
        "source_validation": source,
        "numerical_validation": numerical,
        "figure3_voltage_clamp_validation": figure3["validation"],
        "repeated_pressure_diagnostic": repeated["validation"],
        "claim_boundary": (
            "The exact channel equations and 500-ms saturating-pressure voltage behavior are verified. "
            "The repeated-pressure calculation is a qualitative diagnostic because the channel-only "
            "CellML archive does not encode the original pulse-width protocol."
        ),
    }

    (output_root / "step5_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "figure3_voltage_clamp.json").write_text(
        json.dumps(figure3, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "repeated_pressure_diagnostic.json").write_text(
        json.dumps(repeated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_root / "figure3_voltage_clamp_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(figure3["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(figure3["rows"])

    checksums = {
        path.name: _sha256(path)
        for path in sorted(output_root.iterdir())
        if path.is_file()
    }
    (output_root / "step5_checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
