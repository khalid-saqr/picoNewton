# Step 9 stress-test

## Decision

**PASS — Step 9 is scientifically closed. Proceed to Step 10.**

## Over-simplification

**PASS.** Robustness is not inferred from reciprocal reruns. Nine constitutive paths, exact finite-epsilon block solves, all six arteries, both near-wall conditions, all 89 waveform cases, five endothelial-thickness multipliers and two independent resolutions are evaluated.

## Overcomplication

**PASS.** No new constitutive fit is selected. Path-specific power-law fits are diagnostic only. The final claim remains the Step 8 reciprocal law with a narrower qualifier rather than a higher-dimensional replacement model.

## Feasibility

**PASS.** The general hierarchy reuses the parent one-dimensional harmonic operators. Unit responses are cached by artery and \(\delta\); waveform tests remain small kernel contractions. Exact full-model calculations are limited to the preregistered \(\varepsilon=0.08\) closure.

## Parent-model fidelity

**PASS.** The same operators, centreline regularity, wall no-slip conditions, real-field Lamb construction and two-sided Fourier convention are retained. The beta-only null follows directly from the governing block structure rather than a numerical threshold choice.

## Scientific contribution

**PASS.** Step 9 distinguishes a robust universal waveform-interaction shape from a non-universal constitutive amplitude. This prevents the rank-one result from being overclaimed while preserving its genuine waveform-general content.

## Defects found and corrected

1. The beta-only exact control initially used a relative error against a numerically zero reference. It now has a separate absolute null gate; non-null paths retain the relative closure gate.
2. Quick-profile kernel continuity initially used the publication machine-precision tolerance. It now has a declared reduced-resolution tolerance without changing publication gates.
3. Shape and amplitude errors are exported separately so path-specific kernel norms cannot conceal failure of the frozen reciprocal prefactor.
4. Diagnostic path exponent fits are explicitly prohibited from replacing the Step 8 coefficients.
5. Claim wording is generated as a checksummed artefact and authorises Step 10 only after all robustness gates pass.

## Residual boundaries

- No constitutive amplitude law is supplied for \(\beta\ne\gamma\) or \(\delta\ne1\).
- Robustness paths are controlled tensor perturbations, not measured arterial constitutive data.
- The six-input-harmonic and straight-rigid assumptions remain unchanged.
- Biological thresholds, disease prediction, traction and total radial acceleration remain outside the claim.

## Recommendation

Start Step 10. No Step 9 revision is required. Step 10 should assemble the final Drive-mounted Colab, cold-execute the complete workflow, generate publication figures and tables, and freeze the manuscript-facing archive without changing the locked claim.
