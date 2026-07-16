# Quick-profile execution report

The clean notebook was executed from top to bottom on 2026-07-16 using the
`quick` profile.

## Execution scope

- Six exact v2 arteries and six signed harmonics.
- Verified solver with N=70.
- 256 time points per cycle.
- 64 near-wall quadrature nodes.
- C0-C12 control engine plus the frozen C13 reproduction audit fixture.
- 5 x 5 sensor parameter map for each artery.
- 64 site-stratified physiological Sobol coverage samples.
- Held-out WSS surrogate.
- Eight figures in PDF and PNG.
- CSV, HDF5, run manifest and SHA-256 export.

The isolated build environment did not contain the parent `picoNewton_v2.ipynb`,
so the quick smoke test used the explicit development-only hash skip. The
notebook in the full repository enforces the v2 hash guard; the publication
profile additionally cold-executes an output-stripped v2 copy.

## Notebook integrity

- Cells: 28.
- Code cells: 15.
- Saved code-cell outputs in committed notebook: 0.
- Notebook SHA-256: `59b025f9ee32474661a9e25afc68b0d99086dddb6675e41743e4f307464622db`.

## Runtime verification

| test | observed | threshold | passed |
|---|---:|---:|:---|
| isotropic_analytic | 1.31346e-14 | 1e-08 | True |
| differentiation_polynomial | 2.53575e-13 | 1e-10 | True |
| normalized_backward_residual | 4.9618e-17 | 1e-13 | True |
| sensor_periodic_closure | 1.11022e-16 | 1e-10 | True |
| power_above_h12 | 1.72127e-31 | 1e-12 | True |
| time_force_relative_l2 | 1.69899e-16 | 0.0001 | True |
| near_wall_quadrature_relative_l2 | 1.59844e-06 | 0.0001 | True |

All active-profile numerical guards passed.

## Quick effect gates

| Gate | Result | Observation |
|---|:---:|---|
| E1 | Fail | 3/6 arteries passed in at least 60% of physiological coverage samples; overall 0.656 |
| E2 | Fail | Held-out carotid and brachial residuals remained below 0.005 |
| E3 | Pass | All six arteries contained a connected passing region spanning at least one decade |
| E4 | Pass | 6/6 arteries passed directional specificity in at least 60% of the grid |
| E5 | Fail | 0/6 arteries passed high-harmonic specificity in at least 25% of the grid |
| E6 | Fail | 0/6 arteries passed anisotropy-specific detectability in at least 25% of the grid |
| E7 | Fail | The fifth-percentile effect exceeded 0.005 in 2/6 arteries |
| E8 | Pass | Signed, reversed-direction and magnitude classes were kept separate |

These quick-profile gates are diagnostic only. Publication claims are selected
only from the locked N=150, 2,048-point, 256-quadrature publication run.

## Automated tests

`pytest` passed 15 tests covering analytical validation, differentiation,
hydrodynamic output, sensor bounds and periodicity, deterministic I/O,
checksums, controls, effect-gate evaluation and Sobol admissibility.
