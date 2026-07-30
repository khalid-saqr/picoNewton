from __future__ import annotations

from pathlib import Path
from typing import Any

from .robustness_case import (
    build_hierarchy_cache,
    evaluate_constitutive_paths,
    evaluate_eta_robustness,
    evaluate_exact_closure,
    evaluate_resolution_robustness,
    reconstruct_reciprocal,
)
from .robustness_core import Step9Config
from .robustness_reporting import close_step9
from .robustness_support import frozen_law, validate_step8


def run_robustness_study(
    output_root: str | Path,
    step8_root: str | Path,
    config: Step9Config | None = None,
) -> dict[str, Any]:
    config = config or Step9Config()
    config.validate()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    step8 = validate_step8(step8_root)
    law, frozen_kernel, frozen_parameters = frozen_law(Path(step8["root"]))

    hierarchy_cache, hierarchy_residual = build_hierarchy_cache(config)
    continuity, kernels = reconstruct_reciprocal(
        config, hierarchy_cache, frozen_kernel, frozen_parameters
    )
    path_frame, scale_frame, prediction_frame, archive_arrays = (
        evaluate_constitutive_paths(
            config,
            hierarchy_cache,
            frozen_kernel,
            frozen_parameters,
            kernels,
        )
    )
    exact_frame, exact_residual = evaluate_exact_closure(config, kernels)
    eta_frame = evaluate_eta_robustness(
        config, hierarchy_cache, frozen_kernel, frozen_parameters
    )
    resolution_frame, resolution_residual = evaluate_resolution_robustness(
        config, kernels
    )
    maximum_residual = max(hierarchy_residual, exact_residual, resolution_residual)
    result = close_step9(
        output_root,
        config,
        law,
        continuity,
        path_frame,
        scale_frame,
        prediction_frame,
        exact_frame,
        eta_frame,
        resolution_frame,
        archive_arrays,
        maximum_residual,
    )
    return {
        **result,
        "continuity": continuity,
        "path_summary": path_frame,
        "finite_epsilon": exact_frame,
        "eta": eta_frame,
        "resolution": resolution_frame,
    }
