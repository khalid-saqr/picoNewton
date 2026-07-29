from __future__ import annotations

from typing import Any

import numpy as np

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
from .kernel_workflow_support import closure_row, complex_columns, selection_controls


def analyse_case(
    case: Any,
    config: Step5Config,
    step4_archive: Any | None,
) -> dict[str, Any]:
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

    result: dict[str, Any] = {
        "closure": [],
        "spectra": [],
        "kernels": [],
        "pairs": [],
        "dominant": [],
        "selection": [],
        "step4": [],
        "asymptotic": [
            {
                "artery_id": case.artery_id,
                "artery_name": case.name,
                "epsilon": config.exact_epsilon,
                "scaled_exact_vs_second_order_kernel_relative_l2": relative_l2(
                    exact_excess_kernel / config.exact_epsilon**2,
                    perturbation_kernel,
                ),
            }
        ],
        "arrays": {},
    }

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
        result["closure"].append(
            closure_row(
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
        result["arrays"][f"{case.artery_id}__{kernel_type}__frequencies"] = output_frequencies
        result["arrays"][f"{case.artery_id}__{kernel_type}__spectrum"] = predicted_spectrum
        result["arrays"][f"{case.artery_id}__{kernel_type}__waveform_n"] = np.real(
            kernel_waveform
        )
        result["arrays"][f"{case.artery_id}__{kernel_type}__kernel"] = kernel

        for q, predicted, direct in zip(
            output_frequencies, predicted_spectrum, direct_spectrum, strict=True
        ):
            row = {
                "artery_id": case.artery_id,
                "artery_name": case.name,
                "kernel_type": kernel_type,
                "q": int(q),
            }
            row.update(complex_columns("kernel", predicted))
            row.update(complex_columns("direct", direct))
            result["spectra"].append(row)

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
                row.update(complex_columns("kernel", kernel[i, j]))
                row.update(complex_columns("native_ordered_contribution", ordered[i, j]))
                result["kernels"].append(row)

        by_q: dict[int, list[dict[str, Any]]] = {}
        for item in combine_unordered_pairs(frequencies, ordered):
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
            row.update(complex_columns("combined_contribution", contribution))
            result["pairs"].append(row)
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
                result["dominant"].append(
                    {
                        "artery_id": case.artery_id,
                        "artery_name": case.name,
                        "kernel_type": kernel_type,
                        "q": q,
                        "rank": rank,
                        "m": item["m"],
                        "n": item["n"],
                        "combined_contribution_abs": item["combined_contribution_abs"],
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
        result["step4"].append(
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
        out_q, out_spectrum, _ = evaluate_kernel(frequencies, kernel, phase_canonical)
        phase_waveform = reconstruct_spectrum(out_q, out_spectrum, config.time_points)
        result["closure"].append(
            closure_row(
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
        result["selection"].extend(
            selection_controls(
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
        result["selection"].extend(
            selection_controls(
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
    return result
