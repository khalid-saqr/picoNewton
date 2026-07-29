# Step 6 publication-resolution validation record

## Execution

The publication profile used radial order 150, 2,048 time points, 256 near-wall quadrature nodes, six pressure harmonics and all six native arteries. It consumed and checksum-validated the Step 4 validity domains and the Step 5 exact and second-order kernel archive.

## Gate results

| Gate | Result |
|---|---|
| Step 4 and Step 5 gates consumed | PASS |
| Six arteries complete | PASS |
| Step 5 exact and second-order continuity | PASS |
| Dimensional reconstruction | PASS |
| Harmonic Parseval closure | PASS |
| Pressure-scale separation | PASS |
| Exact validation throughout each Step 4 domain | PASS |
| Inverse-formula verification | PASS |
| Frozen 1 and 10 pN benchmarks | PASS |
| Reachability and constitutive-admissibility states | PASS |
| No silent extrapolation | PASS |

Maximum same-process Step 5 continuity error was \(5.38\times10^{-14}\). Isolated-process exact-archive reproduction remained below \(3.7\times10^{-9}\), passing the separate \(10^{-8}\) portability gate. Maximum Parseval error was \(1.97\times10^{-16}\), and maximum pressure-scale invariance error was \(2.51\times10^{-16}\).

## Exact susceptibility validation

The maximum full-model versus second-order errors were:

| \(\varepsilon\) | Waveform | RMS | Peak |
|---:|---:|---:|---:|
| 0.04 | 0.160% | 0.160% | 0.160% |
| 0.08 | 0.639% | 0.638% | 0.638% |
| 0.10 | 0.998% | 0.997% | 0.997% |

The formal gate respects the artery-specific Step 4 domains: \(\varepsilon=0.10\) is claim-bearing only for femoral and brachial, while the common six-artery domain ends at 0.08.

## Native susceptibility and dimensional response

The dimensionless RMS susceptibility ranks:

\[
\text{brachial} > \text{carotid} > \text{femoral} > \text{iliac} >
\text{thoracic aorta} > \text{aortic root}.
\]

The dimensional second-order RMS coefficient ranks differently:

\[
\text{aortic root} > \text{thoracic aorta} > \text{carotid} >
\text{femoral} > \text{iliac} > \text{brachial}.
\]

Thus intrinsic dimensionless susceptibility and dimensional picoNewton-scale response are not interchangeable. The native force scale reverses the dimensionless ordering.

At \(\varepsilon=0.10\), the second-order predicted RMS excess spans 0.0185–0.1748 pN. The coefficient relative to the isotropic signed-force RMS is 0.982–1.069 per \(\varepsilon^2\), reproducing the approximately one-percent native increment when multiplied by \(0.10^2\).

## Inverse verification

Exact RMS targets generated at \(\varepsilon=0.04\) and 0.08 were inverted independently. The maximum perturbative critical-anisotropy error was 0.321%, and the maximum refined full-model crossing error was \(1.12\times10^{-8}\) in \(\varepsilon\).

## Preregistered force benchmarks

No 1 pN or 10 pN anisotropic-excess benchmark is reached inside any artery's strict validated domain.

For the primary RMS metric, the formal 1 pN estimates are:

| Artery | \(\varepsilon_{\rm crit}^{(2)}\) | Validated maximum | Exact RMS at maximum |
|---|---:|---:|---:|
| Aortic root | 0.239 | 0.08 | 0.113 pN |
| Thoracic aorta | 0.325 | 0.08 | 0.061 pN |
| Femoral | 0.449 | 0.10 | 0.050 pN |
| Carotid | 0.391 | 0.08 | 0.042 pN |
| Iliac | 0.503 | 0.08 | 0.025 pN |
| Brachial | 0.736 | 0.10 | 0.019 pN |

For peak force, the formal 1 pN estimates range from 0.131 to 0.482 and are likewise outside every validated domain.

Seven formal 10 pN estimates satisfy \(\varepsilon\ge1\) and are marked constitutively inadmissible because reciprocal positive definiteness with \(\delta=1\) requires \(|\varepsilon|<1\). All remaining 10 pN estimates are still outside the validated perturbative domain.

## Result boundary

Step 6 establishes native susceptibility functionals and the inverse predictor. It does not yet separate vessel and waveform effects through the two crossed six-by-six matrices, perform harmonic ablations, test broader phase families, fit a reduced law, or assess nonreciprocal and diagonal-viscosity robustness.
