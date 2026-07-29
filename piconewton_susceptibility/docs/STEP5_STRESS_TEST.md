# Step 5 stress-test

## Decision

**PASS — Step 5 is scientifically closed. Proceed to Step 6.**

## Over-simplification

**PASS.** The implementation contains both the exact full-model excess kernel and the second-order coefficient kernel. It verifies native and complex synthetic-phase inputs, not only the published real signed coefficients. It retains DC, sum, difference and doubling terms and exports raw and combined pair contributions.

## Overcomplication

**PASS.** Step 5 does not introduce waveform families, crossed artery-waveform matrices, susceptibility normalisation, threshold inversion, low-rank modelling or mechanosensory interpretation. The only controls are the minimum single- and two-tone cases needed to verify selection rules.

## Feasibility

**PASS.** The kernel is assembled from 13 two-sided frequency slots and one-dimensional near-wall quadrature. Its cost is negligible after the unit harmonic response fields are solved. All publication calculations completed at the inherited resolution.

## Parent-model fidelity

**PASS.** Unit responses use the verified parent solver, inherited operators, boundary conditions, dimensional scaling and near-wall control volume. The one-sided/two-sided conversion uses the locked factor-of-two convention.

## Scientific contribution value

**PASS.** The result moves beyond FFT description. It gives an exact input-output law and identifies which input pairs create each output harmonic. The kernel-level 1% agreement between scaled exact and second-order operators links the exact law directly to the perturbative coefficient required for susceptibility.

## Six arteries

**PASS.** Every native artery is represented in exact and second-order kernels, spectral closure, pair tables and Step 4 continuity. The dominant low-order interaction pattern is shared, while contribution magnitudes and secondary rankings remain artery-dependent.

## Defects prevented or corrected

1. The factor-of-two conversion is applied to coefficients, not hidden inside the kernel.
2. Negative frequencies are retained explicitly, preventing loss of difference-frequency terms.
3. Ordered kernel entries and unordered interpretive contributions are exported separately.
4. Synthetic complex phases are tested to prevent accidental dependence on real signed coefficients.
5. The exact excess is formed from two complete kernels, not from a perturbative approximation.
6. Exposure is explicitly excluded from the bilinear law.
7. A kernel-level asymptotic gate was added to prevent waveform agreement from concealing operator disagreement.

## Residual risks

- The exact production kernel is demonstrated at reciprocal \(\varepsilon=0.10\); nonreciprocal robustness remains later work.
- Pairwise absolute shares can exceed or obscure the net output because cancellation is physical. They must not be called energy fractions.
- The current six physiological waveforms contain only signed real coefficients. Broader phase effects require declared synthetic controls in a later experiment stage.

## Recommendation

Start Step 6. No Step 5 revision is required before constructing the waveform susceptibility and critical-anisotropy predictor.
