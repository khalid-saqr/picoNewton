"""Waveform susceptibility for anisotropic Womersley flow."""

from ._analysis_components import (
    constitutive_robustness,
    crossed_matrices,
    harmonic_pair_attribution,
    native_atlas,
    waveform_catalogue,
    waveform_controls,
)
from .core import (
    AnalysisConfig,
    HarmonicHierarchy,
    ResponseBasis,
    SusceptibilityResult,
    critical_anisotropy,
    derive_hierarchy,
    exact_excess_kernel,
    second_order_kernel,
    susceptibility_from_kernel,
)
from .figures import create_figures
from .public_analysis import build_operator_samples, reduced_law_validation, run_analysis

__version__ = "1.0.0"

__all__ = [
    "AnalysisConfig",
    "HarmonicHierarchy",
    "ResponseBasis",
    "SusceptibilityResult",
    "build_operator_samples",
    "constitutive_robustness",
    "create_figures",
    "critical_anisotropy",
    "crossed_matrices",
    "derive_hierarchy",
    "exact_excess_kernel",
    "harmonic_pair_attribution",
    "native_atlas",
    "reduced_law_validation",
    "run_analysis",
    "second_order_kernel",
    "susceptibility_from_kernel",
    "waveform_catalogue",
    "waveform_controls",
]
