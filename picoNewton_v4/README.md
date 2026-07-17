# picoNewton_v4

Standalone computational package for testing whether anisotropic near-wall Lamb forcing and wall shear stress produce distinguishable membrane loading and Piezo1 gating.

## Current implementation status

- Step 1: scientific protocol frozen.
- Step 2: source, licence, parameter and provenance audit completed.
- Step 3: standalone package and Google Colab runtime scaffold completed.
- Step 4: standalone anisotropic Womersley hydrodynamics completed; publication-resolution six-artery reproduction passed.
- Step 5: independent uncoupled Piezo1 source-model implementation and verification completed.
- **Step 6: passive standard-linear-solid membrane–cortex interface completed and validated for separate WSS and signed-force mappings.**
- Hydrodynamic-to-Piezo1 coupling is not implemented yet.

The package does not import `piconewton_v3`.

## Install and test

```bash
python -m pip install -e "./picoNewton_v4[dev]"
pytest picoNewton_v4/tests
piconewton-v4 --repo-root . --smoke
```

## Run Step 4 hydrodynamics

```bash
piconewton-v4 \
  --repo-root . \
  --run-step4 \
  --profile quick \
  --output /tmp/piconewton_v4_step4
```

Use `--profile publication` for `N=150`, `Nt=2048`, and `Nq=256`.

## Run Step 5 Piezo1 verification

```bash
piconewton-v4 \
  --repo-root . \
  --run-step5 \
  --output /tmp/piconewton_v4_step5
```

Step 5 independently implements the exact four-state CellML model, verifies its parameter transcription, conservation and positivity, compares matrix-exponential propagation with DOP853, and reproduces the source model's 500-ms saturating-pressure voltage dependence.

## Run Step 6 membrane interface

Using an existing Step 4 output directory:

```bash
piconewton-v4 \
  --repo-root . \
  --run-step6 \
  --profile quick \
  --hydrodynamic-root /tmp/piconewton_v4_step4 \
  --output /tmp/piconewton_v4_step6
```

Without `--hydrodynamic-root`, Step 6 regenerates the selected Step 4 profile before applying the membrane interface.

Step 6 maps WSS and signed force separately to a common equivalent stress, propagates each pathway through a passive standard-linear-solid model, and exports equivalent areal strain, branch tensions, stored energy and dissipation. The nonnegative force-exposure observable is not used as signed loading.

## Colab

Open `notebooks/picoNewton_v4_colab.ipynb`. It mounts Google Drive, creates a unique persistent runtime directory, runs computation under `/content`, validates the committed CellML assets, executes the full test suite, runs Steps 4–6, and copies outputs to Drive.

## Scientific boundary

Step 4 reproduces hydrodynamic observables. Step 5 verifies an uncoupled Piezo1 channel model. Step 6 validates a reduced-order membrane–cortex interface. None of these steps equates the Lamb-force control-volume quantity with exact wall traction, establishes a unique endothelial load-transfer law, or feeds hydrodynamic forcing into Piezo1.

The immutable protocol is in `configs/protocol.yaml`. The Step 6 model and parameter domain are documented in `docs/STEP6_MEMBRANE_INTERFACE.md` and `configs/membrane.json`. Original Piezo1 CellML sources are committed under `external/cellml/ogiermann_2025/` with attribution and licence information.
