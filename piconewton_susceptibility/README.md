# picoNewton susceptibility successor

This directory contains the installable package and Google Colab bootstrap for the first Scientific Reports successor to:

> K. M. Saqr, *A transverse picoNewton force revealed in anisotropic Womersley flow*, Scientific Reports 16, 12584 (2026), DOI `10.1038/s41598-026-47474-x`.

## Step 2 scope

Step 2 establishes infrastructure only:

- an installable `piconewton-susceptibility` package;
- a fail-closed adapter to the verified hydrodynamic interface in `picoNewton_v3`;
- a machine-readable parent-source registry;
- reuse of the existing Google Drive, manifest, checksum, and atomic-storage utilities;
- a Colab notebook shell that mounts Drive, checks out the successor branch, installs both packages, validates the parent source, and writes a bootstrap manifest.

No perturbative hierarchy, interaction kernel, susceptibility functional, artery calculation, force threshold, figure, or manuscript result is computed in Step 2.

## Installation from the repository root

```bash
python -m pip install -e "./picoNewton_v3"
python -m pip install -e "./piconewton_susceptibility[dev]"
pytest piconewton_susceptibility/tests
```

## Bootstrap locally

```bash
piconewton-susceptibility-bootstrap \
  --repo-root . \
  --storage local \
  --local-root ./piconewton_susceptibility_outputs
```

The command refuses claim-bearing initialization when the pinned parent files do not match the registry. An explicit development skip is available only for isolated software smoke tests and is recorded as non-claim-bearing.

## Colab

Open `notebooks/scirep_waveform_susceptibility_colab.ipynb`. The notebook:

1. mounts Google Drive;
2. clones or updates `khalid-saqr/picoNewton`;
3. checks out `successor/scirep-waveform-susceptibility`;
4. installs `picoNewton_v3` and this package;
5. validates the frozen parent source chain;
6. writes a Step 2 bootstrap manifest and checksum file to Drive;
7. stops before scientific calculations.

## Scientific boundary

Only these parent modules are permitted:

- `piconewton_v3.hydrodynamics` for the verified flow solver;
- `piconewton_v3.types` for frozen hydrodynamic inputs;
- `piconewton_v3.study_io` for generic storage and provenance utilities.

Mechanosensor, membrane, ion-channel, glycocalyx, calcium-current, signalling, and disease modules are prohibited.
