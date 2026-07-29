# Step 3 architecture: parent-model continuity

## Purpose

Step 3 establishes the complete parent-model continuity layer required before perturbative analysis. It consumes a passing Step 2 completion gate and produces a publication-resolution, six-artery dataset from the frozen `picoNewton_v3` hydrodynamic interface.

## Scientific calculations included

For every frozen artery, Step 3 calculates:

1. verified anisotropic flow at `(beta, gamma, delta) = (0.1, 0.1, 1)`;
2. verified isotropic reference at `(0, 0, 1)`;
3. historical reproduction-mode output for lineage only;
4. signed near-wall Lamb-force integral;
5. magnitude-integrated near-wall Lamb exposure;
6. signed anisotropic excess;
7. exposure difference relative to the isotropic reference;
8. RMS, peak, directional-duty, and high-harmonic metrics.

It also calculates isotropic analytical validation, Gromeka-Lamb mechanics closure, published-alpha consistency, and radial/time/quadrature convergence for both total and excess observables.

## Execution profiles

### Publication

- radial order: 150;
- time points: 2,048;
- near-wall quadrature nodes: 256;
- radial checks: 120 and 180;
- time checks: 1,024 and 4,096;
- quadrature checks: 128 and 512.

Only this profile may produce `allowed_next_step: 4`.

### Quick

The quick profile is a diagnostic path. It can exercise all six arteries but cannot close Step 3 because it is not compared to the frozen publication-resolution historical baseline.

## Historical path

The historical reproduction mode is frozen through `data/historical_mode_baseline.json`. It is tested for executable stability but is explicitly marked `lineage_only`. Its output is never averaged with, substituted for, or used to validate the primary verified result.

## Outputs

- `six_artery_continuity.csv`;
- `historical_mode_discrepancy.csv`;
- `convergence.csv`;
- `isotropic_validation.csv`;
- `six_artery_waveforms.npz`;
- `step3_gate.json`;
- `step3_manifest.json`.

## Boundary

Step 3 does not implement the perturbative hierarchy, harmonic-interaction kernel, waveform susceptibility functional, crossed artery-waveform matrices, critical-anisotropy inversion, low-rank reduction, or publication figures.
