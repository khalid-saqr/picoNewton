# Waveform susceptibility in anisotropic Womersley flow

This directory contains the public numerical software and Google Colab notebook
for the Scientific Reports successor to:

> K. M. Saqr, *A transverse picoNewton force revealed in anisotropic Womersley
> flow*, Scientific Reports 16, 12584 (2026),
> DOI `10.1038/s41598-026-47474-x`.

The package derives the reciprocal weak-anisotropy hierarchy, constructs the
exact two-sided harmonic-interaction kernel, evaluates dimensionless waveform
susceptibility, separates vessel and waveform effects, fits the phase-aware
rank-one reduction, evaluates prescribed force references, and tests selected
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

The package metadata declares `piconewton-v3>=0.1.0` as a runtime dependency.
The sibling installation command above ensures that the repository version is
used.

## Run the complete analysis

```bash
piconewton-waveform-susceptibility \
  --output ./waveform_susceptibility_outputs \
  --figure-dpi 600
```

The command writes:

- the six-artery susceptibility atlas;
- both crossed vessel-waveform matrices;
- waveform-removal, sign, and phase controls;
- harmonic-pair attribution;
- leave-one-artery-out rank-one validation;
- selected constitutive robustness results;
- the reusable operator archive;
- six full-width, multi-panel figures in PDF, SVG, and 600 dpi PNG;
- a machine-readable figure manifest with common dimensions and typography.

## Publication figure standard

All six figures use one shared visual system:

- 180 mm two-column width and height no greater than 170 mm;
- 7 pt Arial/Helvetica-compatible sans-serif lettering;
- 8 pt bold lower-case panel labels;
- one-point minimum line width;
- white background and no decorative three-dimensional effects;
- `cividis` for sequential quantities and `RdBu_r` for centred departures;
- shared normalisation wherever panels are directly comparable;
- explicit dimensionless, pN, percent, or anisotropy units;
- vector PDF and SVG output plus 600 dpi PNG output.

These choices follow the current Scientific Reports and Nature Portfolio figure
recommendations for consistent type, legibility after reduction, panel
lettering, white backgrounds, line weight, and vector line art.

## Google Colab

Open:

```text
notebooks/waveform_susceptibility_colab.ipynb
```

The notebook:

1. mounts Google Drive;
2. creates a unique timestamp-and-UUID run directory;
3. clones the repository and records the exact commit;
4. installs the parent model and public susceptibility package;
5. executes the complete publication-resolution analysis;
6. verifies all required tables, arrays, and figure files;
7. writes runtime metadata and SHA-256 checksums;
8. creates a portable ZIP archive and displays the six figures.

By default the notebook uses `main`. Set the Colab environment variable
`PICONEWTON_REF` to a branch, tag, or commit when reproducing another revision.

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
ruff check waveform_susceptibility/src waveform_susceptibility/tests
```
