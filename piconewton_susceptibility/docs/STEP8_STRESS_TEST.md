# Step 8 stress-test

## Decision

**PASS — Step 8 is scientifically closed. Proceed to Step 9.**

## Over-simplification

**PASS.** The selected law preserves the complete two-sided quadratic frequency-mixing rule. Rank one refers to the interaction kernel, not to deletion of output harmonics. All output frequencies \(q=-12,\ldots,12\) remain in the Parseval waveform functional. Native, high-frequency, sparse, broadband and phase-scrambled inputs are all held out and predicted.

## Overcomplication

**PASS.** The final law contains one vessel prefactor, two vessel exponents and one universal singular interaction mode. Rank two and rank three were tested but rejected by parsimony because they did not improve prediction. No statistical population sweep or unrelated endpoint model was introduced.

## Feasibility

**PASS.** Prediction requires only a twelve-component two-sided coefficient vector, one rank-one kernel contraction and a 25-frequency Parseval norm. The radial solver is not required after the universal mode has been identified.

## Scientific contribution

**PASS.** Step 8 converts the Step 7 empirical separability into an explicit law:

\[
\widehat\Phi_1=1.74055\,\alpha^{-2.01167}\eta^{1.95233}\Psi_1(\mathbf g).
\]

The law provides vessel scaling, phase-aware waveform prediction and a mechanistic reason for the six-artery crossed-matrix structure.

## Validation severity

**PASS.** Each artery is excluded from both available near-wall conditions. Prediction is then evaluated for 89 waveform cases. The selected model passes overall, family-wise, maximum-error and ranking gates. The simpler scalar moment is rejected under simultaneous artery and waveform-family holdout.

## Defects avoided

1. Output-frequency truncation was not confused with kernel rank. The output operator is not low rank, whereas the interaction kernel is.
2. The scalar inverse-harmonic moment was not accepted from native-waveform performance alone; single-tone and spectral-slope holdouts expose its nonuniversality.
3. Rank two and rank three were not retained merely because they pass. They add no predictive benefit.
4. Native diagonal validation was not used as the only evidence. The validation set contains 1,068 held-out predictions per rank.
5. The near-wall exponent is identifiable because Step 7 supplied both fixed-\(\eta\) and native-\(\eta\) matrices.

## Residual risks

- The exponents are interpolation laws over the six parent Womersley regimes and two declared near-wall conditions.
- The universal mode has not yet been tested away from reciprocal anisotropy or \(\delta=1\).
- The waveform basis is truncated to six pressure harmonics, although its quadratic response includes output frequencies through \(|q|=12\).
- The off-diagonal artery-waveform transfers remain controlled counterfactuals.

## Recommendation

Start Step 9. It should test robustness to \(\beta\ne\gamma\), selected \(\delta\ne1\) cases, resolution changes and claim wording. It must not refit the Step 8 law after seeing robustness outcomes.
