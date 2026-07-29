# Step 4 publication-resolution validation record

## Execution

The publication profile used radial order 150, 2,048 time points, 256 near-wall quadrature nodes, six pressure harmonics and the six frozen native arteries. Full reciprocal solutions were calculated at \(\varepsilon=0.005,0.01,0.02,0.04,0.06,0.08,0.10\), with sign-reversed parity checks at \(|\varepsilon|=0.08\).

## Gate results

| Gate | Result |
|---|---|
| Step 3 gate and checksums consumed | PASS |
| Step 3 waveform continuity | PASS |
| Six arteries complete | PASS |
| Hierarchy residual \(\le10^{-10}\) | PASS |
| Full-model residual \(\le10^{-10}\) | PASS |
| Reciprocal sign parity | PASS |
| \(U_\theta=O(\varepsilon)\) | PASS |
| \(U_z-U_z^{(0)}=O(\varepsilon^2)\) | PASS |
| \(\Delta F_s=O(\varepsilon^2)\) | PASS |
| Minimum 1% validity domain \(\varepsilon\ge0.04\) | PASS |

The maximum hierarchy residual was \(4.49\times10^{-16}\), the maximum full-model residual was \(5.51\times10^{-17}\), the maximum parity error was \(4.97\times10^{-16}\), and the maximum cross-step waveform discrepancy was \(4.87\times10^{-16}\).

## Measured orders

Across the six arteries:

- azimuthal-field order: 1.000007–1.000095;
- axial-correction order: 1.999821–2.000208;
- signed-force-excess order: 2.000645–2.000723.

These results establish the reciprocal parity structure rather than merely fitting an empirical power law.

## One-percent validity domains

| Artery | Signed-force law | \(U_\theta^{(1)}\) coefficient | \(U_z^{(2)}\) coefficient |
|---|---:|---:|---:|
| Aortic root | 0.08 | 0.10 | 0.10 |
| Thoracic aorta | 0.08 | 0.10 | 0.10 |
| Femoral | 0.10 | 0.10 | 0.10 |
| Carotid | 0.08 | 0.10 | 0.10 |
| Iliac | 0.08 | 0.10 | 0.10 |
| Brachial | 0.10 | 0.10 | 0.10 |

At \(\varepsilon=0.08\), the maximum signed-force waveform error was 0.643%, the maximum RMS error was 0.638%, and the maximum peak error was 0.638%. At \(\varepsilon=0.10\), four arteries cross the strict 1% waveform criterion by only 0.004–0.008 percentage points; this boundary is therefore reported rather than rounded into a universal 0.10 validity claim.

## Scientific result boundary

Step 4 establishes the perturbative hierarchy and artery-specific validity domain. It does not yet identify harmonic pairs, construct the exact interaction kernel, define the final susceptibility functional, cross vessel and waveform cases, or invert for critical anisotropy.
