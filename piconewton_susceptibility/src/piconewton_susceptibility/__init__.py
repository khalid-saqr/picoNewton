"""Infrastructure for the Scientific Reports waveform-susceptibility successor."""

from .source_registry import SourceRegistry, load_source_registry
from .validation import validate_bootstrap_artifacts

__all__ = ["SourceRegistry", "load_source_registry", "validate_bootstrap_artifacts"]
__version__ = "0.1.1"
