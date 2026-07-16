# Step 3 — standalone package and Colab runtime

## Delivered

- standalone `src/` package with no v3 import;
- Drive-mounted Colab notebook;
- unique persistent run directories;
- local `/content` compute directory;
- atomic file synchronization helper;
- environment and checksum provenance;
- original Piezo1 CellML sources and CC BY 3.0 attribution;
- Python 3.11/3.12 CI scaffold;
- quick and publication configuration profiles.

## Step 3 exit gate

A clean runtime must install the package, create a unique runtime directory, parse all committed CellML files, verify the frozen protocol structure, and pass the scaffold test suite.

Hydrodynamic, membrane, and Piezo1 numerical implementations are deliberately not introduced in Step 3.
