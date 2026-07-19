# picoNewton_v4

`picoNewton_v4` is a stand-alone Python package for testing whether anisotropic near-wall hydrodynamic forcing produces mechanosensory responses distinct from wall shear stress across six arteries.

The package combines:

1. six-artery anisotropic Womersley hydrodynamics;
2. wall shear stress, signed transverse Lamb force, and nonnegative Lamb-force exposure as separate observables;
3. vector-resolved normal and tangential membrane-cortex mechanics;
4. passive fast and slow viscoelastic branches;
5. apical and junctional Piezo1 domains;
6. Piezo1 open probability and current, plus an explicitly exploratory calcium-scale proxy;
7. matched-load, surrogate, anisotropy, harmonic, artery-specificity, and elastic-limit controls;
8. validation, provenance, checksum, table, waveform, and figure generation.

## Historical Colab notebook

The solved notebook is retained unchanged at:

```text
notebooks/picoNewton_v4_colab.ipynb
```

It is the archived first full run and pins the historical package commit. It should not be used as the entry point for the corrected scientific assessment because leaving it unchanged necessarily leaves that commit pin unchanged.

## Corrected scientific entry point

Install the package and run:

```bash
piconewton-v4-study \
  --package-root . \
  --profile full \
  --calibration configs/literature_calibration.json \
  --output /tmp/piconewton_v4_scientific_run
```

This command:

- runs the numerical model in a new output directory;
- uses Piezo1 current as the primary hypothesis endpoint;
- writes calcium to a separate exploratory screen;
- records whether each pathway increases or decreases current magnitude;
- audits signed-force/exposure aggregate degeneracy;
- audits pressure clipping;
- archives the raw WSS and picoNewton force waveforms;
- writes explicit completion gates and SHA-256 checksums.

The command never modifies the historical notebook or an earlier run.

## Installation from a local checkout

```bash
cd picoNewton_v4
python -m pip install -e ".[dev]"
python -m pytest
```

## Original numerical workflow

The lower-level workflow remains available for numerical development:

```bash
piconewton-v4 \
  --package-root . \
  --profile quick \
  --calibration configs/literature_calibration.json \
  --output /tmp/piconewton_v4_quick
```

For the high-resolution calculation, use `--profile full`. The optional legacy localization/channel-count scan can be enabled with `--run-scan`; it is a coarse uncalibrated diagnostic and is not a publication decision engine.

## Primary decision rule

Piezo1 current is the primary endpoint. A hypothesis passes only when the RMS current difference exceeds the predeclared current threshold in at least the required number of arteries. Calcium cannot rescue a failed current decision. Calcium remains available only as an exploratory output until it is independently calibrated against endothelial measurements.

## Scientific observables

The workflow preserves three distinct physical inputs:

- wall shear stress, `tau_w(t)` in Pa;
- signed transverse Lamb force, `F_signed(t)` in N;
- nonnegative Lamb-force exposure, `F_exposure(t)` in N.

Exposure is never used as a signed load. The membrane state preserves normal and tangential mechanics and produces separate apical and junctional tension and pressure histories before Piezo1 gating is calculated.

## Numerical profiles

| Profile | Radial order | Time points | Near-wall nodes | Intended use |
|---|---:|---:|---:|---|
| `quick` | 48 | 256 | 48 | software verification and development |
| `full` | 150 | 2048 | 256 | high-resolution scientific analysis |

## Corrected study outputs

A corrected run produces:

```text
model_outputs/
assessment/
hydrodynamics/
scientific_study_manifest.json
```

Important assessment files include:

- `primary_current_decisions.csv`;
- `exploratory_calcium_screen.csv`;
- `primary_pathway_directionality.csv`;
- `signed_exposure_degeneracy_audit.csv`;
- `pressure_clipping_audit.csv`;
- `scientific_assessment.json`.

The raw physical archive contains WSS, signed-force, exposure, isotropic controls, and anisotropy increments when available.

## Calibration and claim boundary

`configs/literature_calibration.json` is a literature-constrained reference parameterization. Several quantities remain cross-cell or model proxies. The software therefore keeps `claims_enabled` false. A positive biological claim additionally requires independently calibrated endpoint and transfer parameters, a current-based cross-artery result, resolved pressure clipping, identifiable force-class observability, held-out control survival, and independent review.

A current-negative result is a valid completed outcome and is recorded as `negative_under_current_parameterization`. Parameters must not be retuned after observing the result merely to manufacture a pass.

## Documentation

- `docs/SCIENTIFIC_COMPLETION_PATH.md`
- `docs/SCIENTIFIC_MODEL.md`
- `docs/COLAB_EXECUTION.md`
- `docs/OUTPUT_REFERENCE.md`
- `docs/REPRODUCIBILITY.md`

## Source foundations

The hydrodynamic ground truth is based on the six-artery anisotropic Womersley study associated with DOI `10.1038/s41598-026-47474-x`. Piezo1 transition kinetics are represented with an explicit four-state continuous-time Markov model. Model equations and assumptions are documented in the package.
