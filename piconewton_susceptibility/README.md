# picoNewton susceptibility successor

This directory contains the installable package and Google Colab bootstrap for the first Scientific Reports successor to:

> K. M. Saqr, *A transverse picoNewton force revealed in anisotropic Womersley flow*, Scientific Reports 16, 12584 (2026), DOI `10.1038/s41598-026-47474-x`.

## Step 2 scope

Step 2 establishes infrastructure only:

- an installable `piconewton-susceptibility` package;
- a fail-closed adapter to the verified hydrodynamic interface in `picoNewton_v3`;
- a machine-readable parent-source registry;
- reuse of the existing Google Drive, manifest, checksum, and atomic-storage utilities;
- a Colab notebook that mounts Drive, checks out the successor branch, installs both packages, validates the parent source, validates the selected storage backend, writes the Step 2 records, and independently reopens them before authorizing Step 3 as the next workflow stage.

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

The command refuses claim-bearing closure when the pinned parent files do not match the registry. An explicit development skip is available only for isolated software smoke tests and cannot authorize Step 3.

## Completion gate

The Step 2 bootstrap is closed only when all of the following pass:

1. parent-source Git-blob validation;
2. parent API origin and verified-mode restrictions;
3. create/write/rename/read/delete round-trip on the selected storage backend;
4. bootstrap manifest and runtime-validation creation;
5. SHA-256 closure over all final records;
6. independent reopening and verification of the final artifact set.

The authoritative decision record is:

```text
bootstrap/step2/completion_gate.json
```

It contains `allowed_next_step: 3` only for a source-validated, runtime-validated, checksum-closed run. Scientific calculations remain unauthorized inside Step 2.

## Colab

Open `notebooks/scirep_waveform_susceptibility_colab.ipynb`. In Colab the notebook:

1. mounts Google Drive;
2. clones or updates `khalid-saqr/picoNewton`;
3. checks out `successor/scirep-waveform-susceptibility`;
4. installs `picoNewton_v3` and this package;
5. validates the frozen parent source chain;
6. performs a real storage round-trip on the mounted Drive path;
7. writes the manifest, source validation, runtime validation, completion gate, and checksum file;
8. independently reopens and verifies the final records;
9. stops before scientific calculations.

A hosted Drive authentication failure or filesystem failure therefore stops the notebook and cannot produce a passing completion gate.

## Automated validation

The GitHub Actions workflow `.github/workflows/susceptibility-step2.yml` runs on Python 3.10, 3.11, and 3.12. It performs Ruff, compilation, package tests, and a complete local-kernel execution of the notebook. The actual Colab notebook additionally validates the mounted Drive backend during the user-authenticated run.

## Scientific boundary

Only these parent modules are permitted:

- `piconewton_v3.hydrodynamics` for the verified flow solver;
- `piconewton_v3.types` for frozen hydrodynamic inputs;
- `piconewton_v3.study_io` for generic storage and provenance utilities.

Mechanosensor, membrane, ion-channel, glycocalyx, calcium-current, signalling, disease, surrogate, Sobol, and figure-generation modules are prohibited.
