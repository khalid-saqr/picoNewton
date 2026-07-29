from __future__ import annotations

from typing import Any

import numpy as np

from .kernel_core import relative_l2
from .susceptibility_core import (
    ExactNativeEvaluator,
    Step6Config,
    alpha_for_case,
    critical_epsilon_second_order,
    eta_for_case,
    parseval_rms,
    rms,
    scale_invariance_error,
    second_order_native,
    susceptibility_metrics,
)
from .susceptibility_support import isotropic_normalizers


def analyse_case(
    case: Any,
    config: Step6Config,
    step5_archive: Any | None,
    force_valid_max: float,
) -> dict[str, Any]:
    second = second_order_native(case, config)
    force2 = np.asarray(second["waveform_n_per_epsilon2"], dtype=float)
    spectrum2 = np.asarray(second["spectrum"], dtype=complex)
    output_frequencies = np.asarray(second["output_frequencies"], dtype=int)
    scale_n = float(second["force_scale_n"])
    phi_waveform = force2 / scale_n
    phi_spectrum = spectrum2 / scale_n
    metrics = susceptibility_metrics(force2, scale_n)
    normalizers = isotropic_normalizers(case, config)
    force2_rms = rms(force2)
    force2_peak = float(np.max(np.abs(force2)))
    parseval = parseval_rms(phi_spectrum)

    native = {
        "artery_id": case.artery_id,
        "artery_name": case.name,
        "radius_m": case.radius_m,
        "alpha": alpha_for_case(case),
        "eta": eta_for_case(case),
        "pressure_gradient_scale_pa_per_m": case.pressure_gradient_scale_pa_per_m,
        "force_scale_n": scale_n,
        "force2_rms_n_per_epsilon2": force2_rms,
        "force2_peak_abs_n_per_epsilon2": force2_peak,
        "predicted_rms_at_epsilon_0p1_n": 0.1**2 * force2_rms,
        "predicted_peak_at_epsilon_0p1_n": 0.1**2 * force2_peak,
        "phi_parseval_rms": parseval,
        "phi_parseval_relative_error": abs(parseval - metrics["phi_rms"])
        / max(metrics["phi_rms"], 1e-30),
        "signed_excess_to_isotropic_rms_per_epsilon2": force2_rms
        / max(normalizers["isotropic_signed_rms_n"], 1e-30),
        "signed_excess_to_isotropic_peak_per_epsilon2": force2_peak
        / max(normalizers["isotropic_signed_peak_abs_n"], 1e-30),
        "signed_excess_to_wss_force_rms_per_epsilon2": force2_rms
        / max(normalizers["isotropic_wss_force_rms_n"], 1e-30),
        "signed_excess_to_wss_force_peak_per_epsilon2": force2_peak
        / max(normalizers["isotropic_wss_force_peak_abs_n"], 1e-30),
        "max_hierarchy_residual": second["max_residual"],
        **metrics,
        **normalizers,
    }

    arrays = {
        f"{case.artery_id}__time_cycle": np.arange(config.time_points, dtype=float)
        / config.time_points,
        f"{case.artery_id}__phi2_waveform": phi_waveform,
        f"{case.artery_id}__phi2_spectrum": phi_spectrum,
        f"{case.artery_id}__force2_waveform_n": force2,
        f"{case.artery_id}__output_frequencies": output_frequencies,
    }

    total_mean_square = float(np.sum(np.abs(phi_spectrum) ** 2))
    harmonics = [
        {
            "artery_id": case.artery_id,
            "artery_name": case.name,
            "q": int(q),
            "phi_q_real": float(np.real(value)),
            "phi_q_imag": float(np.imag(value)),
            "phi_q_abs": float(np.abs(value)),
            "phi_q_phase_rad": float(np.angle(value)) if abs(value) > 0 else 0.0,
            "mean_square_fraction": float(abs(value) ** 2 / max(total_mean_square, 1e-30)),
        }
        for q, value in zip(output_frequencies, phi_spectrum, strict=True)
    ]

    continuity: dict[str, Any] | None = None
    if step5_archive is not None:
        waveform_key = f"{case.artery_id}__second_order__waveform_n"
        spectrum_key = f"{case.artery_id}__second_order__spectrum"
        if waveform_key not in step5_archive or spectrum_key not in step5_archive:
            raise RuntimeError(f"Step 5 archive is incomplete for {case.artery_id}")
        continuity = {
            "artery_id": case.artery_id,
            "artery_name": case.name,
            "force2_waveform_relative_l2": relative_l2(
                force2, np.asarray(step5_archive[waveform_key])
            ),
            "force2_spectrum_relative_l2": relative_l2(
                spectrum2, np.asarray(step5_archive[spectrum_key])
            ),
            "dimensional_reconstruction_relative_l2": relative_l2(
                scale_n * phi_waveform, force2
            ),
        }

    scale_rows = []
    for factor in config.pressure_scale_factors:
        scale_rows.append(
            {
                "artery_id": case.artery_id,
                "artery_name": case.name,
                "pressure_scale_factor": factor,
                **scale_invariance_error(case, config, factor),
            }
        )

    evaluator = ExactNativeEvaluator(case, config)
    exact_metrics: dict[tuple[float, str], float] = {}
    exact_rows = []
    for epsilon in config.validation_epsilons:
        q_exact, spectrum_exact, waveform_exact = evaluator.spectrum_and_waveform(epsilon)
        if not np.array_equal(q_exact, output_frequencies):
            raise RuntimeError("output frequency axes disagree")
        predicted = epsilon**2 * force2
        exact_rms = rms(waveform_exact)
        exact_peak = float(np.max(np.abs(waveform_exact)))
        exact_metrics[(epsilon, "rms")] = exact_rms
        exact_metrics[(epsilon, "peak_abs")] = exact_peak
        exact_rows.append(
            {
                "artery_id": case.artery_id,
                "artery_name": case.name,
                "epsilon": epsilon,
                "waveform_relative_l2": relative_l2(predicted, waveform_exact),
                "rms_relative_error": abs(rms(predicted) - exact_rms)
                / max(exact_rms, 1e-30),
                "peak_relative_error": abs(float(np.max(np.abs(predicted))) - exact_peak)
                / max(exact_peak, 1e-30),
                "exact_rms_n": exact_rms,
                "predicted_rms_n": rms(predicted),
                "exact_peak_abs_n": exact_peak,
                "predicted_peak_abs_n": float(np.max(np.abs(predicted))),
                "spectrum_relative_l2": relative_l2(
                    epsilon**2 * spectrum2, spectrum_exact
                ),
                "within_step4_valid_domain": epsilon <= force_valid_max + 1e-15,
            }
        )
        if continuity is not None and np.isclose(epsilon, 0.10):
            exact_waveform_key = f"{case.artery_id}__exact_excess__waveform_n"
            exact_spectrum_key = f"{case.artery_id}__exact_excess__spectrum"
            if (
                exact_waveform_key not in step5_archive
                or exact_spectrum_key not in step5_archive
            ):
                raise RuntimeError(f"Step 5 exact archive is incomplete for {case.artery_id}")
            continuity.update(
                {
                    "exact_epsilon_0p1_waveform_relative_l2": relative_l2(
                        waveform_exact, np.asarray(step5_archive[exact_waveform_key])
                    ),
                    "exact_epsilon_0p1_spectrum_relative_l2": relative_l2(
                        spectrum_exact, np.asarray(step5_archive[exact_spectrum_key])
                    ),
                }
            )
        arrays[f"{case.artery_id}__exact_eps_{epsilon:.3f}_n"] = waveform_exact
        arrays[f"{case.artery_id}__predicted_eps_{epsilon:.3f}_n"] = predicted

    inverse_rows = []
    for truth in config.inversion_verification_epsilons:
        if truth > force_valid_max:
            continue
        target = exact_metrics.get((truth, "rms"), evaluator.metric(truth, "rms"))
        estimate = critical_epsilon_second_order(target, force2_rms)
        status, crossing, maximum = evaluator.refine_crossing(target, "rms", force_valid_max)
        inverse_rows.append(
            {
                "artery_id": case.artery_id,
                "artery_name": case.name,
                "truth_epsilon": truth,
                "target_exact_rms_n": target,
                "perturbative_estimate": estimate,
                "perturbative_relative_error": abs(estimate - truth) / truth,
                "full_model_status": status,
                "full_model_crossing": crossing,
                "full_model_absolute_error": abs(crossing - truth)
                if crossing is not None
                else np.nan,
                "validated_domain_max": force_valid_max,
                "exact_metric_at_domain_max_n": maximum,
            }
        )

    critical_rows = []
    for metric in ("rms", "peak_abs"):
        coefficient = force2_rms if metric == "rms" else force2_peak
        maximum_exact = evaluator.metric(force_valid_max, metric)
        for benchmark_pn in config.force_benchmarks_pn:
            target_n = benchmark_pn * 1e-12
            estimate = critical_epsilon_second_order(target_n, coefficient)
            estimate_in_domain = estimate <= force_valid_max
            estimate_admissible = estimate < 1.0
            if maximum_exact < target_n:
                status = "unreachable_within_validated_domain"
                crossing = None
            else:
                status, crossing, maximum_exact = evaluator.refine_crossing(
                    target_n, metric, force_valid_max
                )
            if not estimate_admissible:
                status = (
                    "unreachable_and_formal_estimate_constitutively_inadmissible"
                    if crossing is None
                    else "full_crossing_found_but_formal_estimate_inadmissible"
                )
            elif not estimate_in_domain and status == "full_model_crossing_found":
                status = "full_crossing_found_but_perturbative_estimate_out_of_domain"
            elif not estimate_in_domain and status == "unreachable_within_validated_domain":
                status = "unreachable_and_perturbative_estimate_out_of_domain"
            critical_rows.append(
                {
                    "artery_id": case.artery_id,
                    "artery_name": case.name,
                    "metric": metric,
                    "primary_metric": metric == "rms",
                    "benchmark_pn": benchmark_pn,
                    "benchmark_n": target_n,
                    "coefficient_n_per_epsilon2": coefficient,
                    "perturbative_epsilon_critical": estimate,
                    "validated_domain_max": force_valid_max,
                    "perturbative_estimate_in_domain": estimate_in_domain,
                    "formal_estimate_constitutively_admissible": estimate_admissible,
                    "exact_metric_at_domain_max_n": maximum_exact,
                    "exact_metric_at_domain_max_pn": maximum_exact * 1e12,
                    "full_model_crossing": crossing,
                    "relative_prediction_error": abs(estimate - crossing) / crossing
                    if crossing is not None
                    else np.nan,
                    "status": status,
                }
            )

    return {
        "native": [native],
        "harmonics": harmonics,
        "exact": exact_rows,
        "scale": scale_rows,
        "inverse": inverse_rows,
        "critical": critical_rows,
        "continuity": [continuity] if continuity is not None else [],
        "arrays": arrays,
    }
