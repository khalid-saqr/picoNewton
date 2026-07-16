# Reproducibility protocol

## Immutable baselines

- Repository default branch baseline at workflow design: commit
  `10c958e77e471758876451552b6d25ca7e2b94a0`.
- `picoNewton_v2.ipynb` expected Git blob SHA:
  `9d61c237cda75df338ce0383038f7765c886f503`.
- Exact six-artery harmonic inputs are stored in
  `data/v2_harmonic_inputs.csv`.

## Required publication guards

1. Verify the v2 Git blob SHA.
2. Write an output-stripped copy without changing source cells.
3. Cold-execute the stripped v2 notebook in the publication profile.
4. Pass analytical isotropic Womersley validation.
5. Pass polynomial differentiation and normalized residual checks.
6. Enforce positive dissipation:
   \(\delta-[(\beta+\gamma)/2]^2>0\).
7. Reconstruct real fields before nonlinear multiplication in verified mode.
8. Close the sensor cycle to within `1e-10`.
9. Record configuration, code commit, environment and checksums.
10. Generate figures exclusively from exported source tables.

## Locked publication resolution

- Chebyshev order: 150.
- Time points per cycle: 2,048.
- Near-wall quadrature nodes: 256.
- Sobol samples: 4,096.
- Random seed: `20260716`.

Each claimed effect must be at least ten times its propagated numerical
uncertainty. The thresholds in `configs/effect_gates.json` must not be changed
after publication outputs are inspected.

## Solver modes

`verified` is the scientific production path. `reproduction` exists to show the
impact of the public executable layout and harmonic-product ordering. Both are
reported; they are never mixed in one curve or summary statistic.

## Dataset integrity

Every generated file is registered in `run_manifest.json` with byte count and
SHA-256. The complete run receives a deterministic run ID based on canonical
configuration JSON and the code commit.
