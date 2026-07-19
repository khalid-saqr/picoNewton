# Google Colab execution

## Free-tier design

The notebook is designed for a standard Google Colab CPU runtime.

Safeguards include:

- `quick` profile as the default;
- no GPU requirement;
- optional sensitivity scan disabled by default;
- one artery processed at a time;
- compressed waveform output;
- local `/content` workspace for computational I/O;
- final synchronization to Google Drive;
- explicit cleanup and provenance recording;
- no background services or persistent workers.

## Persistent runtime layout

The notebook mounts Google Drive at `/content/drive` and creates:

```text
/content/drive/MyDrive/picoNewton_v4_runtime/
```

Every execution creates a unique UTC timestamp:

```text
/content/drive/MyDrive/picoNewton_v4_runtime/runs/20260719_143208_UTC/
```

The run directory contains:

```text
configs/
inputs/
logs/
checkpoints/
tables/
figures/
validation/
results/
provenance/
environment/
```

No previous run is overwritten.

## GitHub installation

The notebook configuration cell defines:

```python
REPO_URL = "https://github.com/khalid-saqr/picoNewton.git"
REPO_REF = "main"
PACKAGE_SUBDIR = "picoNewton_v4"
```

The notebook clones the selected ref into `/content`, records the resolved commit SHA, and installs:

```bash
python -m pip install -e ".[dev]"
```

Set `REPO_REF` to a branch, tag, or commit that contains the package.

## Profiles

`quick` is intended for routine free-tier execution.

`full` uses the complete radial, temporal, and near-wall resolution declared in the package. It remains CPU-compatible but takes longer and creates larger output files.

## Recovery after disconnection

Every completed stage writes its metadata to the timestamped Drive directory. A disconnected Colab session does not erase completed Drive outputs. Restarting the notebook creates a new run directory; it never changes an earlier run.
