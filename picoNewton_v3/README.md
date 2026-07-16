# picoNewton v3: mechanosensory observability workflow

This directory contains the third picoNewton study workflow. It preserves the
arterial inputs, anisotropic Womersley equations, Lamb-vector definition and
endothelial control-volume force used by `picoNewton_v2.ipynb`, then adds a
parametric two-state mechanosensor, physiological coverage design, controls,
numerical verification, Google Drive/local I/O, publication figures and a
complete dataset export.

The primary notebook is:

```text
notebooks/picoNewton_v3_mechanosensory.ipynb
```

## Scientific scope

The notebook tests a **standalone generalized mechanosensory-force hypothesis**.
It does not identify an endothelial receptor and does not equate the Lamb
vector with pressure, membrane tension or exact wall traction.

The verified hydrodynamic path reconstructs real velocity and vorticity fields
before evaluating the nonlinear product

\[
\boldsymbol{\ell}=\mathbf{u}\times\boldsymbol{\omega}.
\]

Two solver modes are exposed:

- `verified`: polynomial-tested Chebyshev differentiation and real-field
  nonlinear multiplication. All primary results use this mode.
- `reproduction`: traceability to the current public executable layout. It is
  never sufficient by itself to support a manuscript conclusion.

## Execution profiles

| Profile | Radial order | Points/cycle | Near-wall nodes | Purpose |
|---|---:|---:|---:|---|
| `quick` | 70 | 256 | 64 | End-to-end smoke test and schema validation |
| `publication` | 150 | 2048 | 256 | Full control matrix, figures and dataset |

The publication profile performs the `picoNewton_v2.ipynb` hash guard and is
configured to cold-execute an output-stripped copy of v2 before enabling the
mechanosensory analysis.

## Colab and Google Drive

Open the notebook in Colab and select the `quick` profile first. The notebook:

1. clones the public repository when the source tree is not already present;
2. mounts Google Drive in Colab when storage mode is `auto` or `drive`;
3. creates a deterministic run directory;
4. writes atomic checkpoints, manifests and SHA-256 checksums;
5. exports an archive-ready publication bundle.

Set `PICONEWTON_V3_ROOT` to override the output root. Outside Colab, `auto`
falls back to a local directory.

## Local installation

```bash
python -m pip install -e "./picoNewton_v3[dev]"
pytest picoNewton_v3/tests
```

Run a programmatic quick workflow:

```bash
python picoNewton_v3/run_workflow.py --profile quick --storage local
```

## Publication dataset

Each run contains:

```text
runs/<run_id>/
├── fields/
├── figures/
├── logs/
├── provenance/
├── spectra/
└── summaries/
```

The publication bundle includes configurations, environment metadata, source
and licence manifests, HDF5 signals, tidy CSV tables, figure source data and
checksums. Google Drive is an execution backend, not the final public archive.

## Preliminary Step-6 result

The reduced dry implementation passed all end-to-end software tests, but not
all biological/model-discrimination gates. Connected detectable regions and
directional specificity were observed; the stricter cross-artery detectability,
held-out WSS-surrogate, high-harmonic and anisotropy-specific gates did not pass.
The notebook recomputes every gate at the locked publication resolution and
reports failed gates without changing thresholds.

## Directory structure

```text
picoNewton_v3/
├── configs/                 immutable run and gate configurations
├── data/                    curated inputs, units and provenance manifests
├── docs/                    mathematical, reproducibility and availability text
├── notebooks/               publication notebook
├── references/              BibTeX references
├── src/piconewton_v3/       tested solver, sensor, I/O and workflow modules
├── tests/                   numerical and end-to-end tests
├── pyproject.toml
└── run_workflow.py
```

## Primary references

- Saqr KM. *A transverse picoNewton force revealed in anisotropic Womersley
  flow*. Scientific Reports. 2026. DOI: `10.1038/s41598-026-47474-x`.
- Wiggins P, Phillips R. *Analytic models for mechanotransduction: gating a
  mechanosensitive channel*. PNAS. 2004. DOI: `10.1073/pnas.0307804101`.
- Chuntharpursat-Bon E, et al. *Sensing of shear stress in vascular endothelial
  cells—from physiology to pathology*. Journal of Cell Science. 2026.
  DOI: `10.1242/jcs.264456`.
- Verhees S, Venkataraman C, Ptashnyk M. *Mathematical modelling of
  mechanotransduction via RhoA signalling pathways*. PLOS Computational
  Biology. 2025. DOI: `10.1371/journal.pcbi.1013305`.
- Jones G, et al. *A physiologically realistic virtual patient database for
  arterial haemodynamics*. International Journal for Numerical Methods in
  Biomedical Engineering. 2021. DOI: `10.1002/cnm.3497`; dataset DOI:
  `10.5281/zenodo.4549764`.

See `references/references.bib` and `data/source_manifest.csv` for the complete
source and licence inventory.
