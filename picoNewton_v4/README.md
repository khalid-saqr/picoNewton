# picoNewton_v4

Standalone computational package for testing whether anisotropic near-wall Lamb forcing and wall shear stress produce distinguishable membrane loading and Piezo1 gating.

## Current implementation status

- Step 1: scientific protocol frozen.
- Step 2: source, licence, parameter and provenance audit completed.
- Step 3: standalone package and Google Colab runtime scaffold completed.
- **Step 4: standalone anisotropic Womersley hydrodynamics completed and publication-resolution six-artery reproduction passed.**
- Membrane mechanics and Piezo1 coupling are not implemented yet.

## Step 4 scope

The standalone hydrodynamic layer computes, for all six published arteries:

- anisotropic axial and azimuthal harmonic velocity fields;
- vorticity and the real-field radial Lamb vector;
- signed endothelial control-volume force;
- published magnitude-type force exposure;
- anisotropic axial wall shear stress;
- isotropic Lamb-force and WSS references;
- anisotropy-specific signed and exposure increments;
- harmonic spectra through nonlinear harmonic 12;
- Gromeka–Lamb closure, analytical isotropic validation and grid convergence.

The package does not import `piconewton_v3`.

## Install and test

```bash
python -m pip install -e "./picoNewton_v4[dev]"
pytest picoNewton_v4/tests
piconewton-v4 --repo-root . --smoke
```

## Run Step 4

```bash
piconewton-v4 \
  --repo-root . \
  --run-step4 \
  --profile quick \
  --output /tmp/piconewton_v4_step4
```

Publication resolution:

```bash
piconewton-v4 \
  --repo-root . \
  --run-step4 \
  --profile publication \
  --output /tmp/piconewton_v4_step4_publication
```

## Colab

Open `notebooks/picoNewton_v4_colab.ipynb`. It mounts Google Drive, creates a unique persistent runtime directory, runs computation under `/content`, validates the committed CellML assets, executes the test suite and copies the Step 4 outputs to Drive.

## Scientific boundary

Step 4 reproduces hydrodynamic observables only. It does not equate the Lamb-force control-volume quantity with wall traction or membrane tension and does not calculate Piezo1 gating.

The immutable Step 1 protocol is in `configs/protocol.yaml`. The Step 2 source and parameter lock is in `configs/source_lock.yaml` and `data/`. Original Piezo1 CellML sources are committed under `external/cellml/ogiermann_2025/` with source attribution and licence information.
