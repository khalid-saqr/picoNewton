# Step 4 stress-test

## Decision

**PASS — Step 4 is scientifically closed. Proceed to Step 5.**

## Over-simplification

**PASS.** The implementation does not infer \(O(\varepsilon^2)\) from total-force differences alone. It derives the coefficient boundary-value problems, reconstructs the second-order Lamb-force term, verifies reciprocal parity, measures three independent asymptotic orders, and determines separate force and field validity domains for every artery.

## Overcomplication

**PASS.** Step 4 adds one focused perturbation layer and reuses the verified parent collocation operators. It does not add phase controls, crossed waveforms, low-rank analysis, nonreciprocal robustness, thresholds, mechanosensors or publication figures. Those remain later stages.

## Feasibility

**PASS.** All new calculations are linear one-dimensional harmonic solves. The publication sweep completed for six arteries and seven epsilon values at the inherited publication resolution. The output volume is compact and checksummed.

## Parent-model fidelity

**PASS.** The coefficient equations use the same \(\mathcal L_0\), \(\mathcal L_1\), centreline conditions, wall conditions, six harmonics, dimensional scales and near-wall control volume as the verified parent model. Full-model validation calls the authoritative parent solver directly.

## Scientific contribution value

**PASS.** Step 3 showed that the constitutive increment is approximately 1% of the isotropic background. Step 4 now explains that increment: reciprocal anisotropy generates an odd azimuthal field, an even axial correction, and an even second-order signed-force excess. The result supplies the mathematically justified coefficient that the later susceptibility functional will use.

## Six arteries

**PASS.** All six arteries exhibit the same asymptotic orders, but their strict 1% force-validity limits are not identical. Four are valid through 0.08; femoral and brachial remain valid through 0.10. This prevents an unjustified universal 0.10 claim.

## Defects found and corrected

1. The validity decision was separated into field-coefficient and force-observable domains; a single scalar gate would conceal distinct remainder behaviour.
2. The signed-force waveform, RMS and peak errors are all required; RMS agreement alone could hide phase-local errors.
3. Sign-reversed reciprocal parity was made a mandatory gate, preventing a numerical fit from substituting for the analytic even/odd structure.
4. Exposure is retained only as a time-domain diagnostic and is explicitly excluded from exact-kernel claims.
5. The 1% boundary is not rounded: values slightly above 1% at \(\varepsilon=0.10\) remain classified as outside the strict domain.
6. Step 4 now closes directly against the Step 3 isotropic and \(\varepsilon=0.10\) waveform archive to prevent cross-step numerical drift.

## Residual risks

- The result is established computationally for the reciprocal path, not for arbitrary \(\beta\), \(\gamma\) and \(\delta\).
- The validity limits are tied to the declared 1% metric and the six native waveforms.
- The second-order coefficient is not yet decomposed into harmonic-pair contributions; that is Step 5.

## Recommendation

Start Step 5. No Step 4 revision is required before implementing the exact harmonic-interaction kernel.
