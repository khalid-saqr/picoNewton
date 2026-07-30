# Waveform susceptibility in anisotropic Womersley flow

This directory contains the public numerical software and Google Colab notebook
for the Scientific Reports successor to:

> K. M. Saqr, *A transverse picoNewton force revealed in anisotropic Womersley
> flow*, Scientific Reports 16, 12584 (2026),
> DOI `10.1038/s41598-026-47474-x`.

The package derives the reciprocal weak-anisotropy hierarchy, constructs the
exact two-sided harmonic-interaction kernel, evaluates dimensionless waveform
susceptibility, separates vessel and waveform effects, fits the phase-aware
rank-one reduction, evaluates prescribed force benchmarks, and tests selected
constitutive perturbations.

## Scientific scope

The implementation retains the parent model assumptions:

- straight, rigid, circular vessel;
- axisymmetric, fully developed pulsatile flow;
- six pressure harmonics;
- the verified anisotropic Womersley solver in `picoNewton_v3`;
- reciprocal weak anisotropy for the primary amplitude law;
- signed transverse near-wall Lamb-force response.

The reciprocal amplitude relation applies to `beta = gamma` with `delta = 1`.
The normalised interaction shape is also evaluated under selected
nonreciprocal and diagonal-viscosity perturbations, but those tensors require a
separate amplitude factor.

The software does not claim a biological activation threshold,
patient-specific prediction, traction equivalence, or validity outside the
declared geometry and waveform class.

## Installation

From the repository root:

```bash
python -m pip install -e "./picoNewton_v3"
python -m pip install -e "./waveform_susceptibility[dev]"
```

## Run the complete analysis

```bash
piconewton-waveform-susceptibility \
  --output ./waveform_susceptibility_outputs
```

The command writes:

- the six-artery susceptibility atlas;
- both crossed vessel-waveform matrices;
- waveform-removal, sign, and phase controls;
- harmonic-pair attribution;
- leave-one-artery-out rank-one validation;
- selected constitutive robustness results;
- the reusable operator archive;
- six manuscript-facing figures in PNG and PDF.

## Google Colab

Open:

```text
notebooks/waveform_susceptibility_colab.ipynb
```

The notebook mounts Google Drive, installs the repository packages, executes
the complete analysis, and stores all tables, arrays, figures, and the analysis
summary in a unique Drive directory.

## Python API

```python
from piconewton_waveform_susceptibility import AnalysisConfig, run_analysis

result = run_analysis(
    "./waveform_susceptibility_outputs",
    AnalysisConfig(radial_order=150, time_points=2048, quadrature_nodes=256),
)
```

## Tests

```bash
pytest waveform_susceptibility/tests
ruff check waveform_susceptibility
```
