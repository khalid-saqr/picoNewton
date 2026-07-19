# Reproducibility contract

Every run must record:

- Git repository URL;
- requested Git ref;
- resolved Git commit SHA;
- package version;
- Python version and platform;
- installed package versions;
- selected numerical profile;
- complete configuration files;
- input artery table;
- run start and completion time;
- output SHA-256 checksums.

The Colab notebook performs these steps automatically.

## Determinism

The core hydrodynamic, membrane, and Piezo1 calculations are deterministic. The workflow does not require random sampling unless a future uncertainty module explicitly requests it.

## Immutable run directories

A timestamped run directory is treated as immutable after completion. New parameters or code require a new run directory and a new provenance record.

## Recommended verification

```bash
python scripts/verify_package.py
python -m pip install -e ".[dev]"
python -m pytest
```

A full analysis run should also confirm:

- six unique artery IDs;
- probability-sum error below `1e-9`;
- no negative state probability below tolerance;
- passive normal and tangential mechanical branches;
- nonnegative exposure traction;
- exposure never used as signed loading;
- all output checksums recorded.
