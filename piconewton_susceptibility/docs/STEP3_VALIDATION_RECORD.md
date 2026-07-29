# Step 3 publication-resolution validation record

## Execution

The Step 3 publication profile was executed with:

- radial order 150;
- 2,048 time points;
- 256 near-wall quadrature nodes;
- six pressure harmonics;
- verified anisotropic and isotropic paths;
- historical reproduction path retained separately;
- all six frozen arteries.

## Gate results

| Gate | Result |
|---|---|
| Six arteries complete | PASS |
| Isotropic analytical validation | PASS |
| Maximum normalized residual \(\le 10^{-10}\) | PASS |
| Published-alpha consistency | PASS |
| Gromeka–Lamb mechanics closure | PASS |
| Anisotropic and isotropic exposure nonnegative | PASS |
| Total-force convergence within 1% | PASS |
| Anisotropic-excess convergence within 1% | PASS |
| Historical baseline regression | PASS |
| Historical/verified path separation | PASS |

Maximum total-observable convergence change was \(3.73\times10^{-5}\). Maximum excess-observable convergence change was \(5.00\times10^{-5}\). Maximum mechanics-closure relative error was below \(1.76\times10^{-16}\).

## Six-artery continuity result

| Artery | Computed \(\alpha\) | Total signed RMS (pN) | Isotropic signed RMS (pN) | Signed anisotropic excess RMS (pN) | Excess/isotropic RMS |
|---|---:|---:|---:|---:|---:|
| Aortic root | 22.0160 | 17.6537 | 17.4772 | 0.176599 | 1.010% |
| Thoracic aorta | 17.6128 | 9.54157 | 9.44573 | 0.0958676 | 1.015% |
| Femoral | 5.87092 | 4.68861 | 4.63862 | 0.0500583 | 1.079% |
| Carotid | 5.13706 | 6.24968 | 6.18353 | 0.0662367 | 1.071% |
| Iliac | 6.60479 | 3.79618 | 3.75628 | 0.0399569 | 1.064% |
| Brachial | 2.93546 | 1.89639 | 1.87841 | 0.0186171 | 0.991% |

The verified anisotropy-induced increment is therefore small relative to the isotropic signed Lamb-force background but is resolved by more than two orders of magnitude beyond the measured numerical convergence error.

## Historical discrepancy

Verified-versus-historical signed-waveform relative differences range from approximately 1.25 to 1.96. This is not treated as a failure of the verified path. It confirms that the historical harmonic-product ordering and real-field nonlinear evaluation are not interchangeable. The historical path is retained only for regression and lineage.

## Result boundary

These are parent-continuity results. They do not establish the perturbative \(\varepsilon^2\) law, interaction kernel, waveform susceptibility, phase attribution, crossed matrices, or critical anisotropy.
