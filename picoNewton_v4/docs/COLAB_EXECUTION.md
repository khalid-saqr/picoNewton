# Google Colab execution

## Production entry point

Use:

```text
notebooks/picoNewton_v4_colab.ipynb
```

The notebook is the complete production workflow. Run all cells once from top to bottom in a standard Colab CPU runtime.

## Resolved source and fixed numerical profile

The notebook clones the repository, checks out the reviewed commit `419892085a270b9d0999e2e692e6eefeb408cd81`, verifies it, and records the resolved SHA in every run. The scientific calculation is fixed at the package `full` profile:

- radial order: 150;
- cardiac-cycle time points: 2,048;
- near-wall integration nodes: 256;
- arteries: 6.

The hydrodynamics are computed once. The parameter ensemble reuses the exact archived full-resolution forcing arrays and re-solves the membrane and Piezo1 dynamics.

## Persistent runtime layout

The notebook mounts Google Drive at `/content/drive` and creates a unique directory:

```text
/content/drive/MyDrive/picoNewton_v4_runtime/runs/YYYYMMDD_HHMMSS_UTC_<suffix>/
```

The directory is created before installation or computation so failure diagnostics survive a runtime disconnection. Existing runs are never overwritten.

A completed run contains:

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

## Execution stages

1. Mount Google Drive.
2. Create local and persistent run directories.
3. Clone the repository and check out `419892085a270b9d0999e2e692e6eefeb408cd81`.
4. Install `picoNewton_v4` with development dependencies.
5. Verify the package import path and run the complete test suite.
6. Execute `scripts/run_colab_production.py`.
7. Run the full current-primary scientific study.
8. Validate required files, artery identities, current threshold, and 2,048-point waveform resolution.
9. Run the 13-scenario parameter ensemble.
10. Generate three complete Nature-compatible multi-panel figures.
11. Archive provenance, environment metadata, and SHA-256 checksums.
12. Copy all artifacts to Drive and verify every checksum.

## Parameter ensemble

The one-at-a-time ensemble varies the following quantities around the frozen reference parameterization:

| Quantity | Values |
|---|---|
| Localization area | 0.5, 3, 10 µm² |
| Force-transfer fraction | 0.1, 0.3, 1.0 |
| Channel count | 1,000, 4,165, 10,000 |
| Apical channel fraction | 0.25, 0.5, 0.75 |
| Fast viscoelastic fraction | 0.25, 0.5, 0.75 |
| Pressure ceiling | 35, 70, 140 mmHg |

The baseline is included once, producing 13 scenarios and 78 artery–scenario rows.

## Figure output

Each complete multi-panel figure is exported as:

- editable PDF;
- editable SVG;
- 600 dpi PNG;
- LZW-compressed 600 dpi TIFF.

The canvas is 183 mm wide and below 170 mm high. Body text is 5–7 pt, panel labels are lowercase bold 8 pt, and lines are at least 1 pt wide.

## Failure handling

Installation, test, scientific, parameter, figure, and archive stages write a structured `logs/failure.json` record on failure. Command logs are mirrored to Drive while the run is active.

Do not rerun individual old bootstrap cells from previous notebooks. A new full execution creates a new immutable run directory.
