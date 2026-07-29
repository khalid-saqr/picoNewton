# Step 7 publication-resolution validation record

## Execution

The publication profile used radial order 150, 2,048 time points, 256 near-wall quadrature nodes, six pressure harmonics and all six vessel and waveform cases. Exact crossed validation used reciprocal \(\varepsilon=0.08\).

## Gate results

| Gate | Result |
|---|---|
| Two crossed matrices with 36 entries each | PASS |
| Twelve native diagonal entries | PASS |
| Step 6 native continuity | PASS |
| All 72 exact crossed validations below 1% | PASS |
| Normalised response residual below \(10^{-10}\) | PASS |
| Raw and logarithmic decompositions close | PASS |
| Six direct and six RMS-matched removals per artery | PASS |
| Sign and phase controls complete | PASS |
| Sign/zero-phase degeneracy audited | PASS |
| Twenty-nine causal waveform families complete | PASS |

The maximum crossed waveform error was 0.652%; the maximum RMS error was 0.650%. The maximum normalised response residual was \(9.34\times10^{-16}\).

## Crossed-matrix structure

Both matrices rank the vessel-response regimes as

\[
\text{brachial}>\text{carotid}>\text{femoral}>\text{iliac}>
\text{thoracic aorta}>\text{aortic root}.
\]

Both rank the transferred waveform shapes as

\[
\text{aortic-root waveform}>\text{thoracic waveform}>\text{carotid waveform}>
\text{femoral waveform}>\text{iliac waveform}>\text{brachial waveform}.
\]

For the raw hydrodynamic matrix, vessel, waveform and interaction fractions are 93.07%, 3.37% and 3.56%. For the raw physiological matrix they are 95.23%, 1.20% and 3.57%.

On a logarithmic scale, the interaction fraction falls to \(1.0\times10^{-5}\) for the hydrodynamic matrix and \(1.0\times10^{-6}\) for the physiological matrix. The result indicates near-multiplicative separability, but Step 7 does not fit the corresponding reduced law.

## Harmonic and phase controls

Removing the fundamental reduces native susceptibility by 68.5–88.8%. After restoring the original input RMS, the reduction remains 41.4–49.0%. The second harmonic is the next most influential direct removal in most arteries.

Sign neutralisation changes susceptibility by 0–1.65%. This small change reflects the limited negative coefficients in the published real waveform representation; it does not imply general phase insensitivity.

Across the declared synthetic phase controls, susceptibility ranges from 77.8% to 113.9% of the native value. The aortic cases show the largest phase range.

## Analytical families

For equal-RMS single tones, susceptibility decreases monotonically from harmonic 1 to harmonic 6. Two-tone and sparse-three-tone families containing the fundamental dominate corresponding high-frequency-only families. For spectral slopes \(0\le s\le2\), increasing low-frequency concentration raises mean susceptibility across the six vessel regimes.

## Result boundary

Step 7 establishes the complete experiment dataset and effect decomposition. It does not yet determine whether a one-to-three-mode or scalar reduced predictor passes held-out validation. That is Step 8.
