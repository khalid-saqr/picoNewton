# Step 5 publication-resolution validation record

## Execution

The publication profile used radial order 150, 2,048 time points, 256 near-wall quadrature nodes, all six pressure harmonics and all six native arteries. The exact anisotropic-excess kernel was evaluated at reciprocal \(\varepsilon=0.10\). The second-order kernel was derived independently from the Step 4 coefficient problems.

## Gate results

| Gate | Result |
|---|---|
| Step 4 gate and checksums consumed | PASS |
| Six arteries complete | PASS |
| Exact native and synthetic-phase closure | PASS |
| Second-order native and synthetic-phase closure | PASS |
| Hermitian symmetry and real reconstruction | PASS |
| Response residual \(\le10^{-10}\) | PASS |
| Single- and two-tone selection rules | PASS |
| Step 4 force-coefficient continuity | PASS |
| Scaled exact versus second-order kernel within 2% | PASS |

Maximum exact-kernel waveform error was \(1.68\times10^{-14}\). Maximum second-order waveform error was \(7.31\times10^{-16}\). Maximum response residual was \(4.57\times10^{-16}\), and Step 4 force-coefficient continuity error was \(1.37\times10^{-14}\).

No output above the numerical zero level appeared outside the analytically permitted single- and two-tone frequency sets. DC, difference, sum and frequency-doubling terms were all nonzero in the declared controls.

## Kernel asymptotic closure

At \(\varepsilon=0.10\),

\[
\frac{\Delta K(0.10)}{0.10^2}
\]

differs from \(K^{(2)}\) by 0.965–1.013% in matrix relative \(L_2\) norm across the six arteries. This independently confirms the Step 4 second-order structure at the interaction-operator level.

## Native interaction structure

For the DC second-order response, the fundamental self-pair \((-1,1)\) accounts for 67.8–88.0% of the sum of absolute pair contributions. The second-harmonic self-pair is next, accounting for approximately 11.1–22.7%.

For output harmonic \(q=1\), the fundamental/second-harmonic difference pair \((-1,2)\) is dominant in every artery, contributing 69.4–93.0% of the absolute pairwise sum. For \(q=2\), the fundamental self-pair \((1,1)\) is dominant, contributing 52.7–84.6%.

These percentages are attribution measures, not kinetic-energy fractions.

## Result boundary

Step 5 establishes the exact signed-force interaction law and pair attribution. It does not yet normalise the second-order response into the final waveform susceptibility functional, cross the six vessel regimes with six waveforms, or invert for critical anisotropy.
