# picoNewton_v4

Standalone package for developing and validating a physical interface between anisotropic near-wall Womersley forcing, endothelial membrane–cortex loading, and Piezo1 gating.

## Step 3 status

This branch implements the package and Google Colab runtime scaffold. Hydrodynamic reproduction, membrane mechanics, and coupled Piezo1 simulations are added in later workflow steps.

## Colab requirements

The notebook at `notebooks/picoNewton_v4_colab.ipynb` mounts Google Drive, creates a unique persistent run directory, clones and installs the package into `/content`, performs local computation, writes persistent manifests and outputs to Drive, and validates the committed CellML references.

## Local installation

```bash
python -m pip install -e "./picoNewton_v4[dev]"
pytest picoNewton_v4/tests
piconewton-v4 --repo-root . --smoke
```

## Scientific protocol

The immutable Step 1 protocol is in `configs/protocol.yaml`. The Step 2 source and parameter lock is in `configs/source_lock.yaml` and `data/`.

## Third-party Piezo1 CellML sources

The original Piezo1 CellML source files are committed under `external/cellml/ogiermann_2025/` under CC BY 3.0, with source attribution, permanent URIs, and revision identifiers.
