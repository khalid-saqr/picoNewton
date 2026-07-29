# Step 6 stress-test

## Decision

**PASS — Step 6 is scientifically closed. Proceed to Step 7.**

## Over-simplification

**PASS.** Step 6 does not reduce susceptibility to a single RMS number. It exports the dimensionless time waveform, complex harmonic coefficients, peak response, positive and negative RMS components, sign-duty fractions, isotropic normalisation and wall-shear-force normalisation. It validates exact waveforms throughout the artery-specific Step 4 domains and tests inversion against reachable full-model targets.

## Overcomplication

**PASS.** The implementation remains restricted to the six native artery-waveform pairs. It does not yet execute the crossed matrices, harmonic removal, phase scrambling, spectral-slope families, low-rank reduction or constitutive robustness. Peak inversion is retained as the declared secondary metric rather than expanded into a broad threshold catalogue.

## Feasibility

**PASS.** The publication workflow completed locally in approximately 22 seconds. Exact validation and inversion use cached one-dimensional harmonic response kernels. Output tables and arrays remain compact and checksummed.

## Parent-model fidelity

**PASS.** The susceptibility is formed from the Step 5 second-order signed-force kernel and normalised by the inherited \(\rho A_{\rm EC}U_*^2\) scale. The exact validator uses the verified reciprocal parent model. The local parent package did not export wall shear, so Step 6 reconstructs the inherited isotropic wall shear directly from the verified harmonic derivative and published dimensional scaling rather than altering the parent solver.

## Scientific contribution value

**PASS.** Step 6 produces two consequential findings.

1. Dimensionless susceptibility and dimensional force rank the arteries differently. Small arteries, especially the brachial and carotid, have the largest \(\Phi_{2,\rm rms}\), while the aortic root has the largest dimensional coefficient because its force scale dominates.
2. The preregistered 1 and 10 pN anisotropic-excess levels are not reachable within the validated reciprocal-anisotropy domain. This separates the parent paper's total picoNewton Lamb exposure from the smaller constitutive excess and prevents them from being conflated.

The inverse law remains useful because it quantifies how far outside the validated domain each prescribed level lies and accurately recovers lower reachable exact targets.

## Six arteries

**PASS.** Every artery contributes native time-domain, harmonic, directional, scale-normalised, exact-validation and inversion results. Artery-specific Step 4 limits are preserved; no universal 0.10 domain is reintroduced.

## Defects found and corrected

1. Wall-shear normalisation initially relied on an output absent from the locally installed parent revision. It is now reconstructed from the verified isotropic harmonic solution.
2. Exact Step 5 continuity was extended to the \(\varepsilon=0.10\) full excess waveform and spectrum, not only the second-order coefficient.
3. The exact-validation gate now follows each artery's Step 4 domain instead of using only a common 0.08 cutoff.
4. Formal estimates with \(\varepsilon\ge1\) are explicitly labelled constitutively inadmissible.
5. Primary manuscript benchmarks are kept separate from lower exact targets used solely to verify the inversion algorithm.
6. Unreachable benchmarks return structured states and no fabricated full-model crossing.
7. Cross-process BLAS reproduction of the exact archive is tested with a separate \(10^{-8}\) portability gate; the algebraic second-order and dimensional closures retain their \(10^{-11}\) gate.

## Residual risks

- The primary 1 and 10 pN levels have no full-model crossing inside the validated domain; their reported critical values are formal extrapolations, not validated predictions.
- The native dimensionless ranking combines \(\alpha\), native \(\eta\) and waveform structure. Step 7 is required to separate these effects.
- Directional signs inherit the parent coordinate convention and must not be translated into biological action.
- The theorem remains restricted to reciprocal anisotropy with \(\delta=1\).

## Recommendation

Start Step 7. No Step 6 revision is required before executing the complete waveform experiment design and the hydrodynamic and physiological crossed six-by-six susceptibility matrices.
