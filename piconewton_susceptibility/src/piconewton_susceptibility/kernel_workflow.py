from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from piconewton_v3 import V2_ARTERY_CASES

from .kernel_core import (
    Step5Config,
    canonical_coefficients,
    combine_unordered_pairs,
    direct_full_waveform,
    direct_second_order_waveform,
    evaluate_kernel,
    full_force_kernel,
    near_wall_basis,
    reconstruct_spectrum,
    relative_l2,
    sampled_spectrum,
    second_order_force_kernel,
    unit_full_response,
    unit_perturbation_response,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_step4_artifacts(root: str | Path) -> dict[str, Any]:
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
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise RuntimeError(f"Step 4 artifact failed checksum validation: {name}")
    return {"passed": True, "gate": gate, "manifest": manifest, "root": str(root)}


def _complex_columns(prefix: str, value: complex) -> dict[str, float]:
    return {
        f"{prefix}_real": float(np.real(value)),
        f"{prefix}_imag": float(np.imag(value)),
        f"{prefix}_abs": float(np.abs(value)),
        f"{prefix}_phase_rad": float(np.angle(value)) if abs(value) > 0.0 else 0.0,
    }


def _closure_row(
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


def _selection_allowed(selected: Sequence[int]) -> set[int]:
    signed = set(selected) | {-value for value in selected}
    return {first + second for first in signed for second in signed}


def _selection_controls(
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
        allowed = _selection_allowed(selected)
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


def run_harmonic_kernel(
    output_root: str | Path,
    step4_root: str | Path,
    config: Step5Config | None = None,
    *,
    require_step4: bool = True,
) -> dict[str, Any]:
    if config is None:
        config = Step5Config()
    config.validate()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if require_step4:
        step4 = _validate_step4_artifacts(step4_root)
    else:
        step4 = {"passed": False, "development_skip": True}

    step4_archive = None
    if require_step4 and config.profile == "publication":
        archive_path = Path(step4["root"]) / "perturbation_waveforms.npz"
        if not archive_path.is_file():
            raise RuntimeError("publication Step 5 requires the Step 4 waveform archive")
        step4_archive = np.load(archive_path)

    closure_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    kernel_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    dominant_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    step4_rows: list[dict[str, Any]] = []
    asymptotic_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}

    for case in V2_ARTERY_CASES:
        perturbation = unit_perturbation_response(case, config)
        anisotropic = unit_full_response(case, config, config.exact_epsilon)
        isotropic = unit_full_response(case, config, 0.0)
        radial, perturbation_nw = near_wall_basis(case, perturbation, config)
        radial_full, anisotropic_nw = near_wall_basis(case, anisotropic, config)
        radial_iso, isotropic_nw = near_wall_basis(case, isotropic, config)
        if not np.array_equal(radial, radial_full) or not np.array_equal(radial, radial_iso):
            raise RuntimeError("near-wall grids disagree")

        frequencies, perturbation_kernel = second_order_force_kernel(case, perturbation_nw)
        frequencies_full, anisotropic_kernel = full_force_kernel(case, anisotropic_nw)
        frequencies_iso, isotropic_kernel = full_force_kernel(case, isotropic_nw)
        if not (
            np.array_equal(frequencies, frequencies_full)
            and np.array_equal(frequencies, frequencies_iso)
        ):
            raise RuntimeError("kernel frequency axes disagree")
        exact_excess_kernel = anisotropic_kernel - isotropic_kernel
        asymptotic_rows.append(
            {
                "artery_id": case.artery_id,
                "artery_name": case.name,
                "epsilon": config.exact_epsilon,
                "scaled_exact_vs_second_order_kernel_relative_l2": relative_l2(
                    exact_excess_kernel / config.exact_epsilon**2,
                    perturbation_kernel,
                ),
            }
        )

        native_one_sided = np.asarray(case.harmonic_coefficients, dtype=complex)
        _, native_coefficients = canonical_coefficients(native_one_sided)

        for kernel_type, kernel, direct_waveform, residual in (
            (
                "second_order",
                perturbation_kernel,
                direct_second_order_waveform(
                    case, perturbation_nw, native_one_sided, config.time_points
                ),
                perturbation.max_residual,
            ),
            (
                "exact_excess",
                exact_excess_kernel,
                direct_full_waveform(
                    case,
                    anisotropic_nw,
                    isotropic_nw,
                    native_one_sided,
                    config.time_points,
                ),
                max(anisotropic.max_residual, isotropic.max_residual),
            ),
        ):
            output_frequencies, predicted_spectrum, ordered = evaluate_kernel(
                frequencies, kernel, native_coefficients
            )
            kernel_waveform = reconstruct_spectrum(
                output_frequencies, predicted_spectrum, config.time_points
            )
            direct_spectrum = sampled_spectrum(direct_waveform, output_frequencies)
            closure_rows.append(
                _closure_row(
                    case.artery_id,
                    case.name,
                    kernel_type,
                    output_frequencies,
                    predicted_spectrum,
                    direct_waveform,
                    kernel_waveform,
                    residual,
                )
            )
            arrays[f"{case.artery_id}__{kernel_type}__frequencies"] = output_frequencies
            arrays[f"{case.artery_id}__{kernel_type}__spectrum"] = predicted_spectrum
            arrays[f"{case.artery_id}__{kernel_type}__waveform_n"] = np.real(kernel_waveform)
            arrays[f"{case.artery_id}__{kernel_type}__kernel"] = kernel

            for q, predicted, direct in zip(
                output_frequencies, predicted_spectrum, direct_spectrum, strict=True
            ):
                row = {
                    "artery_id": case.artery_id,
                    "artery_name": case.name,
                    "kernel_type": kernel_type,
                    "q": int(q),
                }
                row.update(_complex_columns("kernel", predicted))
                row.update(_complex_columns("direct", direct))
                spectrum_rows.append(row)

            for i, m in enumerate(frequencies):
                if m == 0:
                    continue
                for j, n in enumerate(frequencies):
                    if n == 0:
                        continue
                    row = {
                        "artery_id": case.artery_id,
                        "artery_name": case.name,
                        "kernel_type": kernel_type,
                        "m": int(m),
                        "n": int(n),
                        "q": int(m + n),
                    }
                    row.update(_complex_columns("kernel", kernel[i, j]))
                    row.update(_complex_columns("native_ordered_contribution", ordered[i, j]))
                    kernel_rows.append(row)

            combined = combine_unordered_pairs(frequencies, ordered)
            by_q: dict[int, list[dict[str, Any]]] = {}
            for item in combined:
                contribution = complex(item["contribution"])
                q = int(item["q"])
                row = {
                    "artery_id": case.artery_id,
                    "artery_name": case.name,
                    "kernel_type": kernel_type,
                    "m": int(item["m"]),
                    "n": int(item["n"]),
                    "q": q,
                }
                row.update(_complex_columns("combined_contribution", contribution))
                pair_rows.append(row)
                by_q.setdefault(q, []).append(row)
            for q, candidates in by_q.items():
                if q < 0 or not candidates:
                    continue
                ranked = sorted(
                    candidates,
                    key=lambda item: item["combined_contribution_abs"],
                    reverse=True,
                )
                total_abs = sum(item["combined_contribution_abs"] for item in ranked)
                for rank, item in enumerate(ranked[:5], start=1):
                    dominant_rows.append(
                        {
                            "artery_id": case.artery_id,
                            "artery_name": case.name,
                            "kernel_type": kernel_type,
                            "q": q,
                            "rank": rank,
                            "m": item["m"],
                            "n": item["n"],
                            "combined_contribution_abs": item[
                                "combined_contribution_abs"
                            ],
                            "fraction_of_pairwise_absolute_sum": item[
                                "combined_contribution_abs"
                            ]
                            / max(total_abs, 1e-30),
                        }
                    )

        if step4_archive is not None:
            key = f"{case.artery_id}__force2_n"
            if key not in step4_archive:
                raise RuntimeError(f"Step 4 archive is incomplete for {case.artery_id}")
            _, perturbation_spectrum, _ = evaluate_kernel(
                frequencies, perturbation_kernel, native_coefficients
            )
            perturbation_waveform = np.real(
                reconstruct_spectrum(
                    np.arange(-12, 13), perturbation_spectrum, config.time_points
                )
            )
            step4_rows.append(
                {
                    "artery_id": case.artery_id,
                    "artery_name": case.name,
                    "force2_waveform_relative_l2": relative_l2(
                        perturbation_waveform, np.asarray(step4_archive[key])
                    ),
                }
            )

        phase_coefficients = native_one_sided * np.exp(
            1j * np.asarray(config.synthetic_phases_rad)
        )
        _, phase_canonical = canonical_coefficients(phase_coefficients)
        for kernel_type, kernel, direct in (
            (
                "second_order",
                perturbation_kernel,
                direct_second_order_waveform(
                    case, perturbation_nw, phase_coefficients, config.time_points
                ),
            ),
            (
                "exact_excess",
                exact_excess_kernel,
                direct_full_waveform(
                    case,
                    anisotropic_nw,
                    isotropic_nw,
                    phase_coefficients,
                    config.time_points,
                ),
            ),
        ):
            out_q, out_spectrum, _ = evaluate_kernel(
                frequencies, kernel, phase_canonical
            )
            phase_waveform = reconstruct_spectrum(out_q, out_spectrum, config.time_points)
            closure_rows.append(
                _closure_row(
                    case.artery_id,
                    case.name,
                    f"{kernel_type}_synthetic_phase",
                    out_q,
                    out_spectrum,
                    direct,
                    phase_waveform,
                    max(
                        perturbation.max_residual,
                        anisotropic.max_residual,
                        isotropic.max_residual,
                    ),
                )
            )

        if case.artery_id == "aortic_root":
            selection_rows.extend(
                _selection_controls(
                    case,
                    config,
                    "second_order",
                    frequencies,
                    perturbation_kernel,
                    lambda coeffs: direct_second_order_waveform(
                        case, perturbation_nw, coeffs, config.time_points
                    ),
                )
            )
            selection_rows.extend(
                _selection_controls(
                    case,
                    config,
                    "exact_excess",
                    frequencies,
                    exact_excess_kernel,
                    lambda coeffs: direct_full_waveform(
                        case,
                        anisotropic_nw,
                        isotropic_nw,
                        coeffs,
                        config.time_points,
                    ),
                )
            )

    closure = pd.DataFrame(closure_rows)
    spectra = pd.DataFrame(spectrum_rows)
    kernels = pd.DataFrame(kernel_rows)
    pairs = pd.DataFrame(pair_rows)
    dominant = pd.DataFrame(dominant_rows)
    selection = pd.DataFrame(selection_rows)
    step4_continuity = pd.DataFrame(step4_rows)
    asymptotic = pd.DataFrame(asymptotic_rows)

    closure.to_csv(output_root / "kernel_closure.csv", index=False)
    spectra.to_csv(output_root / "force_spectra.csv", index=Falsi
    kernels.to_csv(output_root / "kernel_entries.csv", index=False)
    pairs.to_csv(output_root / "pair_contributions.csv", index=False)
    dominant.to_csv(output_root / "dominant_pairs.csv", index=False)
    selection.to_csv(output_root / "selection_rule_controls.csv", index=Falsi
    asymptotic.to_csv(output_root / "kernel_asymptotic_closure.csv", index=Falsi
    if config.profile == "publication":
        step4_continuity.to_csv(output_root / "step4_kernel_continuity.csv", index=Falsi
    np.savez_compressed(output_root / "kernel_archive.npz", **arrays)

    outside = selection[selection["outside_allowed"]]
    allowed = selection[selection["allowed"]]
    allowed_group = allowed.groupby(["kernel_type", "control", "q"])["relative_to_max"].max()
    required_nonzero = bool((allowed_group > 1e-10).all())
    gates: dict[str, Any] = {
        "step": 5,
        "profile": config.profile,
        "publication_profile": config.profile == "publication",
        "step4_gate_consumed": bool(step4.get("passed")),
        "six_arteries_complete": closure["artery_id"].nunique() == 6,
        "exact_kernel_closure_passed": bool(
            closure[closure["kernel_type"].str.startswith("exact_excess")][
                ["waveform_relative_l2", "spectrum_relative_l2"]
            ].to_numpy().max()
            <= config.closure_tolerance
        ),
        "second_order_kernel_closure_passed": bool(
            closure[closure["kernel_type"].str.startswith("second_order")][
                ["waveform_relative_l2", "spectrum_relative_l2"]
            ].to_numpy().max()
            <= config.closure_tolerance
        ),
        "hermitian_and_real_reconstruction_passed": bool(
            closure[
                ["reconstruction_imaginary_relative_max", "hermitian_relative_max"]
            ].to_numpy().max()
            <= config.closure_tolerance
        ),
        "response_residual_passed": bool(
            closure["max_normalized_response_residual"].max() <= 1e-10
        ),
        "selection_rule_support_passed": bool(
            (outside["relative_to_max"].max() if len(outside) else 0.0)
            <= config.selection_tolerance
            and required_nonzero
        ),
        "kernel_asymptotic_2pct_passed": bool(
            asymptotic["scaled_exact_vs_second_order_kernel_relative_l2"].max()
            <= 0.02
        ),
        "step4_force2_continuity_passed": bool(
            step4_continuity["force2_waveform_relative_l2"].max() <= 1e-12
        )
        if config.profile == "publication"
        else False,
        "dc_sum_difference_and_doubling_present": required_nonzero,
        "exposure_used_in_kernel": False,
        "susceptibility_or_threshold_inversion_run": False,
    }
    required = [
        "publication_profile",
        "step4_gate_consumed",
        "six_arteries_complete",
        "exact_kernel_closure_passed",
        "second_order_kernel_closure_passed",
        "hermitian_and_real_reconstruction_passed",
        "response_residual_passed",
        "selection_rule_support_passed",
        "kernel_asymptotic_2pct_passed",
        "step4_force2_continuity_passed",
        "dc_sum_difference_and_doubling_present",
    ]
    gates["passed"] = all(bool(gates[name]) for name in required)
    (output_root / "step5_gate.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True), encoding="utf-8"
    )

    output_names = [
        "kernel_closure.csv",
        "force_spectra.csv",
        "kernel_entries.csv",
        "pair_contributions.csv",
        "dominant_pairs.csv",
        "selection_rule_controls.csv",
        "kernel_asymptotic_closure.csv",
        "kernel_archive.npz",
        "step5_gate.json",
    ]
    if config.profile == "publication":
        output_names.append("step4_kernel_continuity.csv")
    manifest: dict[str, Any] = {
        "step": 5,
        "status": "complete" if gates["passed"] else "failed",
        "profile": config.profile,
        "step4_gate_consumed": bool(step4.get("passed")),
        "scientific_scope": "exact_signed_harmonic_interaction_kernel_only",
        "exact_epsilon": config.exact_epsilon,
        "allowed_next_step": 6 if gates["passed"] else None,
        "gates": gates,
        "files": {},
    }
    for name in output_names:
        path = output_root / name
        manifest["files"][name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    (output_root / "step5_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "closure": closure,
        "spectra": spectra,
        "kernels": kernels,
        "pairs": pairs,
        "dominant": dominant,
        "selection": selection,
        "asymptotic": asymptotic,
        "step4_continuity": step4_continuity,
    }
