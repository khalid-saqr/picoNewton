from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .kernel_core import (
    Step5Config,
    canonical_coefficients,
    evaluate_kernel,
    sampled_spectrum,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_step4_artifacts(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    gate_path = root / "step4_gate.json"
    manifest_path = root / "step4_manifest.json"
    if not gate_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Step 5 requires Step 4 gate and manifest")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise RuntimeError("Step 5 requires a passing Step 4 gate")
    if manifest.get("status") != "complete" or manifest.get("allowed_next_step") != 5:
        raise RuntimeError("Step 4 manifest does not authorize Step 5")
    for name, record in manifest.get("files", {}).items():
        path = root / name
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise RuntimeError(f"Step 4 artifact failed checksum validation: {name}")
    return {"passed": True, "gate": gate, "manifest": manifest, "root": str(root)}


def complex_columns(prefix: str, value: complex) -> dict[str, float]:
    return {
        f"{prefix}_real": float(np.real(value)),
        f"{prefix}_imag": float(np.imag(value)),
        f"{prefix}_abs": float(np.abs(value)),
        f"{prefix}_phase_rad": float(np.angle(value)) if abs(value) > 0.0 else 0.0,
    }


def closure_row(
    artery_id: str,
    artery_name: str,
    kernel_type: str,
    output_frequencies: np.ndarray,
    predicted_spectrum: np.ndarray,
    direct_waveform: np.ndarray,
    kernel_waveform: np.ndarray,
    max_residual: float,
) -> dict[str, Any]:
    direct_spectrum = sampled_spectrum(direct_waveform, output_frequencies)
    hermitian = max(
        abs(predicted_spectrum[i] - np.conj(predicted_spectrum[-i - 1]))
        for i in range(len(predicted_spectrum))
    ) / max(np.max(np.abs(predicted_spectrum)), 1e-30)
    from .kernel_core import relative_l2

    return {
        "artery_id": artery_id,
        "artery_name": artery_name,
        "kernel_type": kernel_type,
        "waveform_relative_l2": relative_l2(np.real(kernel_waveform), direct_waveform),
        "spectrum_relative_l2": relative_l2(predicted_spectrum, direct_spectrum),
        "reconstruction_imaginary_relative_max": float(
            np.max(np.abs(np.imag(kernel_waveform)))
            / max(np.max(np.abs(np.real(kernel_waveform))), 1e-30)
        ),
        "hermitian_relative_max": float(hermitian),
        "max_normalized_response_residual": float(max_residual),
    }


def selection_allowed(selected: Sequence[int]) -> set[int]:
    signed = set(selected) | {-value for value in selected}
    return {first + second for first in signed for second in signed}


def selection_controls(
    case: Any,
    config: Step5Config,
    kernel_type: str,
    frequencies: np.ndarray,
    kernel: np.ndarray,
    direct_builder: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for control_name, selected, phases in (
        ("single_tone_h2", (2,), (0.41,)),
        ("two_tone_h2_h5", (2, 5), (0.41, -0.73)),
    ):
        one_sided = np.zeros(6, dtype=complex)
        for harmonic, phase in zip(selected, phases, strict=True):
            one_sided[harmonic - 1] = np.exp(1j * phase)
        freq, coefficients = canonical_coefficients(one_sided)
        if not np.array_equal(freq, frequencies):
            raise RuntimeError("frequency axes disagree")
        output_frequencies, spectrum, _ = evaluate_kernel(frequencies, kernel, coefficients)
        waveform = direct_builder(one_sided)
        direct_spectrum = sampled_spectrum(waveform, output_frequencies)
        scale = max(np.max(np.abs(spectrum)), 1e-30)
        allowed = selection_allowed(selected)
        for q, predicted, direct in zip(
            output_frequencies, spectrum, direct_spectrum, strict=True
        ):
            rows.append(
                {
                    "artery_id": case.artery_id,
                    "kernel_type": kernel_type,
                    "control": control_name,
                    "q": int(q),
                    "allowed": int(q) in allowed,
                    "predicted_abs": float(abs(predicted)),
                    "direct_abs": float(abs(direct)),
                    "relative_to_max": float(abs(predicted) / scale),
                    "outside_allowed": int(q) not in allowed,
                }
            )
    return rows
