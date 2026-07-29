# Step 2 stress-test

## Decision

**Status: CONDITIONAL PASS.**

The package and notebook scaffold are implementation-complete and no unresolved code defect has been identified. Step 3 must nevertheless not begin until one clean Google Colab execution confirms Drive mounting, branch checkout, editable installation of both packages, source validation, manifest writing, and checksum writing in the actual hosted runtime.

This is a validation condition, not a request to redesign Step 2.

## Scope tested

Step 2 is restricted to:

- the installable `piconewton-susceptibility` package;
- the frozen parent-source registry;
- a fail-closed adapter to the verified parent hydrodynamic API;
- reuse of existing storage, manifest, checksum, and Google Drive utilities;
- a Colab bootstrap notebook that stops before scientific calculations;
- automated tests for the package, source chain, adapter, bootstrap authorization, and notebook boundary.

The perturbative hierarchy, harmonic interaction kernel, susceptibility functional, six-artery calculations, crossed matrices, critical-anisotropy inversion, and publication figures remain outside Step 2.

## Stress-test findings

### 1. Over-simplification

**Assessment: PASS.**

The step is not merely an empty directory or notebook shell. It establishes the minimum infrastructure needed for Step 3:

- an independently installable package;
- a versioned source registry containing the DOI, repository, source commit, v2 notebook blob, permitted parent modules, exact hydrodynamic-module blobs, six artery inputs, dimensional constants, and publication benchmarks;
- Git-blob verification of the parent source;
- a typed adapter that allows only the verified hydrodynamic configuration;
- deterministic bootstrap manifests and checksums;
- explicit claim-bearing versus development-only authorization states;
- a Colab orchestration path;
- executable tests.

Adding scientific calculations at this stage would violate the locked workflow rather than strengthen Step 2.

### 2. Overcomplication

**Assessment: PASS.**

The implementation does not copy or fork the Womersley solver. It imports the verified parent hydrodynamic surface and reuses `StudyStore` and `resolve_study_root` instead of creating a second Drive, checkpoint, manifest, or checksum framework.

No mechanosensor, membrane, ion-channel, glycocalyx, calcium-current, signalling, disease, surrogate, Sobol, or figure-generation module is used.

### 3. Feasibility

**Assessment: LOCAL PASS; COLAB VALIDATION PENDING.**

Completed local checks:

- editable package installation;
- wheel construction;
- nine automated tests;
- Python source parsing and bytecode compilation;
- notebook parsing as nbformat 4;
- bootstrap-only notebook contract;
- fail-closed parent-source validation behavior;
- non-claim-bearing development bootstrap behavior.

The remaining feasibility check is a cold run in Google Colab. The local environment cannot prove hosted Drive authentication or Colab-specific path behavior.

### 4. Parent-model fidelity

**Assessment: PASS.**

The source registry pins:

- DOI `10.1038/s41598-026-47474-x`;
- repository `khalid-saqr/picoNewton`;
- published-source commit `4c3c36db0578373cc4e48d9d8c7e8a85944ed1cb`;
- `picoNewton_v2.ipynb` blob `9d61c237cda75df338ce0383038f7765c886f503`;
- exact blobs of `hydrodynamics.py`, `types.py`, and `study_io.py`;
- six native arteries and their pressure-gradient scales and harmonic coefficients;
- frozen dimensional constants and 1 pN/10 pN manuscript benchmark set.

The adapter refuses reproduction mode for new results.

### 5. Scientific-boundary integrity

**Assessment: PASS.**

The notebook and bootstrap manifest explicitly state:

- `scientific_calculations_run = false`;
- `scientific_calculations_authorized = false`.

The notebook contract rejects calls to the harmonic solver, full hydrodynamic calculation, interaction kernel, susceptibility functional, or critical-anisotropy calculation.

### 6. Step 3 safety

**Assessment: PASS WITH VALIDATION CONDITION.**

A source-validated bootstrap may record `allowed_next_step = 3`. A development skip is explicitly non-claim-bearing and records `allowed_next_step = null`; it cannot authorize Step 3.

## Defects discovered and corrected during the stress-test

1. **Parent package initializer side effect.** Importing a `piconewton_v3` submodule executes the package initializer, which also loads later sensor code. A global `sys.modules` prohibition would therefore reject the legitimate hydrodynamic import. The adapter was corrected to validate the origin of every exposed symbol and to export no sensor callable.

2. **Development-skip authorization.** The first bootstrap draft could have implied progression after a skipped source check. It was corrected so a development skip is non-claim-bearing and cannot authorize Step 3.

3. **Colab/local repository discovery.** The notebook was corrected to discover the repository root robustly rather than assume one current working directory.

4. **Dimensional-scale ambiguity.** The registry now distinguishes the 10 kPa m^-1 reference normalization from the six artery-specific native pressure-gradient scales.

5. **Step 1 amendment persistence.** The approved Step 1 Amendment 01A is included in the repository tree with its SHA-256 record rather than remaining only as a local export.

## Validation record

| Check | Result |
|---|---|
| Automated tests | 9 passed |
| Package import/version | Passed |
| Registry schema and freeze | Passed |
| Known Git-blob calculation | Passed |
| Missing-parent fail-closed behavior | Passed |
| Parent API origin enforcement | Passed |
| Verified-mode enforcement | Passed |
| Development bootstrap authorization | Passed |
| Notebook parse and bootstrap-only boundary | Passed |
| Wheel build | Passed |
| Ruff | Not executed; executable unavailable in the current environment |
| Clean Google Colab run | Pending |

## Pre-Step 3 recommendation

No additional code correction is presently required. The only required action before Step 3 is to execute the committed notebook in a clean Google Colab runtime and confirm that the three Step 2 artifacts are written to Google Drive with a passing source-validation record.
