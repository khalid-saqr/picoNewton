# Step 8 publication-resolution validation record

## Decision

**PASS — a rank-one universal interaction law is selected.**

## Selected law

The full-data fit is

\[
\widehat\Phi_1(\alpha,\eta,\mathbf g)
=1.74055\,\alpha^{-2.01167}\eta^{1.95233}\Psi_1(\mathbf g).
\]

The first singular mode retains 99.9986% of the full universal-kernel energy. Across the six leave-one-artery-out folds, the minimum retained energy is 99.9979%.

## Held-out prediction

The rank-one law was tested on 1,068 artery-waveform predictions:

| Metric | Result | Limit |
|---|---:|---:|
| Median relative error | 2.262% | 5% |
| 90th-percentile relative error | 10.443% | 12% |
| Maximum relative error | 15.933% | 20% |
| Minimum native ranking Spearman correlation | 1.000 | 0.95 |

The median error for every waveform family is below 2.86%. The largest family maximum is 15.93%.

| Family | Median | Maximum |
|---|---:|---:|
| Native arterial | 2.659% | 15.367% |
| Phase challenge | 2.237% | 15.245% |
| Single tone | 2.041% | 14.082% |
| Sparse three tone | 2.421% | 14.420% |
| Spectral slope | 2.860% | 15.933% |
| Two tone | 2.187% | 14.762% |

## Vessel scaling

The fitted exponents are stable under artery deletion:

\[
p_\alpha\in[-2.03562,-1.99686],\qquad
p_\eta\in[1.94633,1.96117].
\]

The leave-one-artery-out vessel-scale error has median 2.10%, 90th percentile 5.19% and maximum 7.99%.

## Model selection

Ranks one, two and three all pass. Rank two and rank three do not improve the error distribution: their median errors are 2.449% and 2.424%, compared with 2.262% for rank one. Rank one is therefore selected by parsimony.

The inverse-harmonic scalar moment is rejected. Under simultaneous artery and waveform-family holdout it has median error 6.93%, 90th-percentile error 22.25% and maximum error 48.81%. Its single-tone median error is 25.05%, demonstrating that a phase-blind energy moment is not waveform-general.

## Interpretation

The Step 7 near-multiplicative matrix structure is explained by two independent reductions:

1. the vessel amplitude follows approximately \(\alpha^{-2}\eta^2\);
2. the normalised harmonic interaction kernel is effectively rank one.

The result is a phase-aware predictive functional, not merely a fitted artery ranking. It predicts native, sparse, broadband, high-frequency and synthetic-phase waveforms without re-solving the radial boundary-value problem.

## Boundary

The law is established only within the parent straight-rigid reciprocal model and the sampled \((\alpha,\eta)\) domain. Step 9 must test nonreciprocal and diagonal-viscosity robustness before the claim is locked.
