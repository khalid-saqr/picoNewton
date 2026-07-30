# Step 10 architecture — final publication workflow

## Purpose

Step 10 is an assembly and reproducibility step. It does not derive, fit or modify scientific laws. It consumes passing, checksum-validated Steps 2–9 and the locked Step 9 claim, builds the manuscript-facing output set, and freezes a single publication archive.

## Cold workflow

The final Colab notebook creates a unique Google Drive run directory, clones the repository, checks out the pinned Step 10 implementation commit, installs the parent and successor packages, and executes Steps 2–10 in order. Every step must produce a passing gate before the next command runs.

## Main-paper figures

The six frozen main figures are:

1. parent-model continuity and perturbative hierarchy;
2. exact interaction law and kernel equivalence;
3. waveform amplitude, sign and phase controls;
4. susceptibility landscape and perturbative validity;
5. six-artery physiological susceptibility atlas;
6. crossed susceptibility matrix and critical-anisotropy prediction.

Step 8 model selection and Step 9 constitutive/numerical robustness are exported as supplementary figures.

## Archive contents

`publication_archive.zip` contains:

- every manifest-tracked output from Steps 2–9;
- six main figures in PNG and PDF;
- supplementary figures;
- figure-source CSV files;
- main and supplementary tables;
- the frozen reduced law and claim lock;
- environment, Git and workflow provenance;
- an internal publication inventory and SHA-256 list.

The external `publication_archive.sha256` and `step10_manifest.json` authenticate the completed ZIP.

## Fail-closed rules

Step 10 fails when any prior gate, manifest or checksum fails; when the Step 9 claim is not locked; when the reduced law or claim changes during copying; when any main figure or source table is missing; or when the archive checksum cannot be generated.

No scientific refit, constitutive extension, biological endpoint model or claim edit is permitted in Step 10.
