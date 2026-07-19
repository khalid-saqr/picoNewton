# picoNewton_v4

`picoNewton_v4` is a stand-alone Python package and Google Colab workflow for testing whether anisotropic near-wall hydrodynamic forcing produces mechanosensory responses that are distinct from wall shear stress across six arteries.

The package combines:

1. six-artery anisotropic Womersley hydrodynamics;
2. wall shear stress, signed transverse Lamb force, and nonnegative Lamb-force exposure as separate observables;
3. vector-resolved normal and tangential membrane-cortex mechanics;
4. fast and slow passive viscoelastic branches;
5. apical and junctional Piezo1 domains;
6. Piezo1 open probability, current, and calcium-scale endpoints;
7. direct, matched-load, surrogate, anisotropy, harmonic, artery-specificity, and elastic-limit comparisons;
8. validation, provenance, checksum, table, waveform, and figure generation.

## Primary entry point

Open the notebook in Google Colab:

```text
notebooks/picoNewton_v4_colab.ipynb
```

The notebook is written as a complete scientific document. It mounts Google Drive, creates a unique UTC date-time-stamped run directory, clones and installs the package from GitHub, records the Git commit, runs tests, executes the selected numerical profile, generates outputs, and copies all final artifacts to Drive.

## Google Drive directory structure

Every notebook execution creates a new directory under:

```text
MyDrive/picoNewton_v4_runtime/runs/YYYYMMDD_HHMMSS_UTC/
```

Each run contains:

```text
configs/
inputs/
logs/
checkpoints/
tables/
figures/
validation/
results/
provenance/
environment/
```

Previous runs are never overwritten.

## Installation from a local checkout

```bash
cd picoNewton_v4
python -m pip install -e ".[dev]"
python -m pytest
```

Expected package test result:

```text
12 passed
```

## Command-line execution

Quick profile:

```bash
piconewton-v4 \
  --package-root . \
  --profile quick \
  --calibration configs/literature_calibration.json \
  --output /tmp/piconewton_v4_quick
```

Full-resolution profile:

```bash
piconewton-v4 \
  --package-root . \
  --profile full \
  --calibration configs/literature_calibration.json \
  --output /tmp/piconewton_v4_full
```

The optional localization/channel-count scan is disabled by default because it is substantially more expensive. Enable it with `--run-scan`.

## Scientific observables

The workflow preserves three distinct inputs:

- wall shear stress, `tau_w(t)` in Pa;
- signed transverse Lamb force, `F_signed(t)` in N;
- nonnegative Lamb-force exposure, `F_exposure(t)` in N.

The exposure observable is never used as a signed load.

The membrane state preserves normal and tangential mechanics and produces separate apical and junctional tension and pressure histories before Piezo1 gating is calculated.

## Numerical profiles

| Profile | Radial order | Time points | Near-wall nodes | Intended use |
|---|---:|---:|---:|---|
| `quick` | 48 | 256 | 48 | Colab free-tier verification and development |
| `full` | 150 | 2048 | 256 | final high-resolution analysis |

The notebook defaults to `quick`. Change one configuration variable to run `full`.

## Main outputs

A completed run produces:

- `six_artery_summary.csv`
- `hypothesis_effects.csv`
- `loao_vector_surrogates.csv`
- `artery_feature_distances.csv`
- `waveforms.npz`
- `validation.json`
- `manifest.json`
- optional `hypothesis_decisions.csv`
- standard PNG figures
- environment and Git provenance
- SHA-256 checksums

See `docs/OUTPUT_REFERENCE.md` for field definitions.

## Calibration boundary

`configs/literature_calibration.json` is a literature-constrained reference parameterization. Several quantities remain cross-cell or model proxies. The configuration is suitable for reproducible computational screening and sensitivity analysis, but scientific inference must preserve those uncertainty labels and use predeclared thresholds.

## Documentation

- `docs/SCIENTIFIC_MODEL.md`
- `docs/COLAB_EXECUTION.md`
- `docs/OUTPUT_REFERENCE.md`
- `docs/REPRODUCIBILITY.md`

## Source foundations

The hydrodynamic ground truth is based on the six-artery anisotropic Womersley study associated with DOI `10.1038/s41598-026-47474-x`. Piezo1 transition kinetics are reproduced as an explicit four-state continuous-time Markov model. All model equations and assumptions used by the notebook are documented in the package.
