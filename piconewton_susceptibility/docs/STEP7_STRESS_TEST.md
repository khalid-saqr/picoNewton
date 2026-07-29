# Step 7 stress-test

## Decision

**PASS — Step 7 is scientifically closed. Proceed to Step 8.**

## Over-simplification

**PASS.** Step 7 contains both crossed matrices, exact validation of every matrix entry, native continuity, direct and RMS-matched harmonic removals, sign controls, deterministic phase controls and analytical equal-RMS families. The six arteries are used as both vessel regimes and waveform sources rather than only as a final validation set.

## Overcomplication

**PASS.** The programme is finite and causal: 72 crossed entries, 24 native controls per artery and 29 analytical waveform families per vessel. No random population sweep, machine learning, mechanosensor, low-rank fit or constitutive robustness study has been introduced.

## Feasibility

**PASS.** Unit harmonic responses are solved once per vessel and reused for both matrices and all waveforms. Subsequent calculations are small kernel contractions. The complete publication calculation fits comfortably within the existing one-dimensional solver architecture.

## Parent-model fidelity

**PASS.** Step 7 reuses the Step 4 perturbation hierarchy and Step 5 exact and second-order kernel definitions. It changes only the waveform coefficients and the declared near-wall ratio. Pressure-scale amplitudes are removed from the dimensionless matrices and retained only for the native Step 6 continuity comparison.

## Scientific contribution value

**PASS.** The matrices provide the first direct separation of vessel and waveform effects. Vessel response dominates raw variance, while the nearly vanishing logarithmic interaction indicates a possible multiplicative reduced law. Fundamental-content and phase controls explain why equal-RMS waveforms need not produce equal susceptibility.

## Six arteries

**PASS.** Every artery appears in all six rows and all six columns of both matrices. All native diagonal cases reproduce Step 6. No artery is omitted from exact validation, ablations, phase controls or analytical families.

## Defects found and corrected

1. Solver residuals were initially divided only by the right-hand-side norm. They now use the parent solver's normalised backward error.
2. Harmonic removal alone confounded spectral redistribution with reduced input RMS. Six RMS-restored removal controls were added for every artery.
3. Sign neutralisation and zero-phase alignment are identical for the real signed native coefficients. Their degeneracy is now explicitly audited rather than counted as two independent results.
4. Vessel and waveform main-effect tables were added; variance fractions alone were insufficient for traceable ranking.
5. Exact validation was applied to every off-diagonal entry, not only the native diagonal.

## Residual risks

- The off-diagonal cases are controlled counterfactual transfers, not physiological observations.
- The near-multiplicative structure is a dataset result, not yet a validated reduced law.
- Phase sensitivity is assessed with declared synthetic controls because full measured complex phases are unavailable.
- Constitutive robustness beyond reciprocal \(\delta=1\) remains outside Step 7.

## Recommendation

Start Step 8. No Step 7 revision is required before testing low-rank and compact waveform reductions with held-out validation.
