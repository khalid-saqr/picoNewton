# picoNewton_v4

`picoNewton_v4` is a stand-alone Python and Google Colab workflow for testing whether anisotropic near-wall hydrodynamic forcing produces Piezo1 responses distinct from wall shear stress across six arteries.

The package combines:

1. six-artery anisotropic Womersley hydrodynamics;
2. wall shear stress, signed transverse Lamb force, and nonnegative force exposure as separate observables;
3. vector-resolved normal and tangential membrane–cortex mechanics;
4. passive fast and slow viscoelastic branches;
5. apical and junctional Piezo1 domains;
6. Piezo1 open probability and current, plus an explicitly exploratory calcium-scale proxy;
7. direct, matched-load, surrogate, anisotropy, harmonic, artery-specificity, and elastic-limit controls;
8. a predeclared full-resolution parameter ensemble;
9. validation, provenance, checksums, raw arrays, tables, and Nature-compatible multi-panel figures.

## Primary Google Colab entry point

Open and run:

```text
notebooks/picoNewton_v4_colab.ipynb
```

The notebook is distributed without saved outputs and is designed for a standard Google Colab CPU runtime. It:

- mounts Google Drive;
- creates a unique UTC-stamped directory under `MyDrive/picoNewton_v4_runtime/runs/`;
- clones the merged `main` branch and records the resolved commit SHA;
- installs the package and development dependencies;
- runs the complete package test suite;
- solves the `full` numerical profile: radial order 150, 2,048 time points, and 256 near-wall nodes;
- runs `scripts/run_colab_production.py`, which executes the corrected current-primary scientific assessment;
- reuses the archived full-resolution hydrodynamics for a 13-scenario membrane–Piezo1 parameter ensemble;
- exports three complete multi-panel figures as PDF, SVG, 600 dpi PNG, and LZW-compressed 600 dpi TIFF;
- records environment metadata, Git provenance, scientific outputs, and SHA-256 checksums.

Every run is immutable and previous runs are never overwritten.

## Parameter ensemble

The notebook uses a deterministic one-at-a-time design around the literature-constrained reference parameterization. It varies:

- force localization area;
- force-transfer fraction;
- channel count;
- apical channel fraction;
- fast viscoelastic fraction;
- pressure ceiling.

The ensemble is diagnostic. It does not convert proxy quantities into experimental calibration and it does not enable claims automatically.

## Nature-compatible figures

The notebook follows the current Nature figure specifications used for the generated files:

- 183 mm double-column width;
- height below 170 mm;
- standard sans-serif typeface;
- 5–7 pt body text;
- bold lowercase 8 pt panel labels;
- minimum 1 pt line width;
- white background;
- complete multi-panel figures arranged as single files;
- vector PDF and SVG plus 600 dpi PNG and TIFF.

Figure legends are written to `figures/figure_legends.txt`.

## Package command-line entry points

The corrected current-primary study remains available outside Colab:

```bash
piconewton-v4-study \
  --package-root . \
  --profile full \
  --calibration configs/literature_calibration.json \
  --output /tmp/piconewton_v4_scientific_run
```

The lower-level numerical workflow is available as:

```bash
piconewton-v4 \
  --package-root . \
  --profile quick \
  --calibration configs/literature_calibration.json \
  --output /tmp/piconewton_v4_quick
```

## Primary decision rule

Piezo1 current is the primary endpoint. A hypothesis passes only when its RMS current difference exceeds the predeclared current threshold in at least the required number of arteries. Calcium is written to a separate exploratory screen and cannot rescue a failed current decision.

## Outputs

A completed notebook run contains:

```text
scientific_study/
parameter_study/
figures/
logs/
provenance/
environment/
FINAL_MANIFEST.json
SHA256SUMS.json
```

Important outputs include:

- `scientific_study/model_outputs/six_artery_summary.csv`;
- `scientific_study/model_outputs/hypothesis_effects.csv`;
- `scientific_study/assessment/primary_current_decisions.csv`;
- `scientific_study/assessment/exploratory_calcium_screen.csv`;
- `scientific_study/assessment/scientific_assessment.json`;
- `scientific_study/hydrodynamics/physical_forcing_waveforms.npz`;
- `parameter_study/parameter_scenarios.csv`;
- `parameter_study/parametric_artery_results.csv`;
- `parameter_study/parameter_scenario_summary.csv`;
- `parameter_study/parametric_current_waveforms.npz`;
- complete publication figures and legends.

## Calibration and claim boundary

`configs/literature_calibration.json` is literature constrained. Several quantities remain cross-cell or model proxies. The software keeps `claims_enabled` false. A positive biological claim requires independently calibrated endpoint and transfer parameters, a current-based cross-artery result, resolved pressure clipping, identifiable force-class observability, held-out control survival, and independent review.

A current-negative result is a valid completed outcome. Parameters must not be retuned after observing the result merely to manufacture a pass.

## Documentation

- `docs/COLAB_EXECUTION.md`
- `docs/SCIENTIFIC_COMPLETION_PATH.md`
- `docs/SCIENTIFIC_MODEL.md`
- `docs/OUTPUT_REFERENCE.md`
- `docs/REPRODUCIBILITY.md`

## Source foundations

The hydrodynamic ground truth is based on the six-artery anisotropic Womersley study associated with DOI `10.1038/s41598-026-47474-x`. Piezo1 transition kinetics are represented with an explicit four-state continuous-time Markov model.
