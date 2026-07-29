# Step 2 architecture and implementation boundary

## Objective

Step 2 creates the installable successor package and the Colab/Google Drive bootstrap. It does not implement or execute the perturbative hierarchy, harmonic-interaction kernel, susceptibility functional, six-artery calculations, crossed matrices, critical-anisotropy inversion, or publication figures.

## Parent interface

The successor imports only:

- `piconewton_v3.hydrodynamics`;
- `piconewton_v3.types`;
- `piconewton_v3.study_io`.

The adapter rejects a new-results configuration unless `mode="verified"`. Because Python executes a package initializer before loading a submodule, the enforceable boundary is the origin of every callable and type exposed by the adapter; no mechanosensory callable is exported or used.

## Source-chain validation

The registry pins:

- parent DOI;
- repository and published-source commit;
- `picoNewton_v2.ipynb` blob;
- hydrodynamics, types, and generic storage module blobs;
- the 10 kPa m^-1 reference normalisation, the six artery-specific native pressure-gradient scales, frozen dimensional constants, and the 1 pN/10 pN publication benchmark set.

Validation uses Git's canonical blob identity: `SHA1(b"blob <length>\\0" + file_bytes)`.

## Storage reuse

No new storage engine is introduced. The successor calls the existing `resolve_study_root` and `StudyStore` interfaces. Step 2 writes only:

```text
bootstrap/step2/
├── bootstrap_manifest.json
├── source_validation.json
└── checksums.sha256
```

The manifest records whether source validation passed, whether a development skip was used, and that no scientific calculations were authorized or executed.

## Colab behavior

The notebook mounts Drive, clones or updates the public repository, checks out the successor branch, installs the parent and successor packages, runs the source-validated bootstrap, displays the manifest, and stops.

## Step 3 entry condition

Step 3 may begin only after:

- package installation succeeds;
- the source registry validates;
- all Step 2 tests pass;
- the notebook contains no scientific calculation cells;
- the stress-test reports no unresolved blocker.
