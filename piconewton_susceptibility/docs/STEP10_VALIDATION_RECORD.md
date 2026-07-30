# Step 10 validation record

## Local implementation checks

The Step 10 modules were compiled and exercised through a cold synthetic workflow fixture containing independently checksummed Steps 2–9. The fixture used the real Step 9 publication table schemas and verified:

- direct Step 2 validation;
- Steps 3–9 gate and manifest validation;
- exact preservation of the Step 8 law and Step 9 claim lock;
- six main figures in PNG and PDF;
- three supplementary figures;
- eighteen figure-source CSV files;
- five main and six supplementary tables;
- environment and provenance records;
- internal checksums;
- ZIP creation and external SHA-256 verification;
- a passing terminal Step 10 manifest.

The fixture is a software validation only and is not a scientific result archive.

## Production validation path

The final Drive notebook and GitHub workflow execute the complete publication-resolution chain. Production closure is attained only when `step10_gate.json` records `passed: true`, `step10_manifest.json` records `workflow_complete: true`, and the SHA-256 value in `publication_archive.sha256` matches `publication_archive.zip`.

## Locked scientific boundary

Step 10 retains the Step 9 primary statement and qualifier exactly. It does not re-estimate the rank, prefactor, Womersley exponent, near-wall exponent or constitutive robustness limits.
