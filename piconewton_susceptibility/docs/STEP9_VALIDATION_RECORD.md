# Step 9 publication-resolution validation record

## Decision

**PASS — the reciprocal law is retained with a constitutive-amplitude restriction.**

## Frozen-law continuity

Independent reconstruction returned

\[
C=1.74054784,\qquad p_\alpha=-2.01167284,\qquad p_\eta=1.95232848,
\]

and rank-one energy 0.99998604, reproducing Step 8 without refitting.

## Finite-epsilon closure

All nine constitutive paths were compared with the exact full block system at \(\varepsilon=0.08\). The maximum non-null hierarchy-kernel error was 0.7976%, below the 2% gate. The beta-only exact and perturbative excesses remained numerically zero.

## Constitutive-shape robustness

Across eight non-null paths, twelve vessel/near-wall operators and 89 waveforms, the frozen singular interaction mode produced:

| Metric | Result | Limit |
|---|---:|---:|
| Median shape error | 0.653% | 5% |
| Maximum shape error | 10.996% | 20% |
| Minimum path rank-one energy | 99.9981% | 99.9% |

The largest shape error occurred for the gamma-only path. Moderate beta-gamma asymmetry and \(\delta\in\{0.8,1.2\}\) therefore preserve the phase-aware interaction structure.

## Amplitude restriction

The frozen reciprocal amplitude prefactor is not constitutively universal. Maximum uncorrected amplitude errors were:

- beta-low: 73.0%;
- gamma-low: 168.6%;
- gamma-only: 334.4%;
- \(\delta=0.8\): 23.0%;
- \(\delta=1.2\): 30.8%.

The final claim is therefore restricted to \(\beta=\gamma\) and \(\delta=1\). Nonreciprocal or diagonal-viscosity-shifted tensors require a separate constitutive amplitude factor.

## Vessel-exponent diagnostics

The fitted path exponents were used only as diagnostics. Across non-null paths,

\[
p_\alpha\in[-2.07787,-2.00197],\qquad
p_\eta\in[1.94919,1.95375].
\]

The near-wall exponent is especially stable, while the gamma-only path produces the largest Womersley-exponent drift.

## Near-wall and numerical robustness

For native \(\eta\) multiplied by 0.8–1.2, the frozen reciprocal law had median, 90th-percentile and maximum errors of 1.713%, 5.787% and 8.998%.

The independent resolution pairs changed representative susceptibilities by at most

\[
1.03\times10^{-6}.
\]

The maximum backward residual was \(8.27\times10^{-16}\).

## Locked scientific statement

Within the straight, rigid, axisymmetric six-harmonic model and the tested \((\alpha,\eta)\) domain, reciprocal weak constitutive anisotropy produces a signed transverse near-wall force susceptibility represented by a frozen rank-one, phase-aware waveform functional with vessel scaling approximately \(\alpha^{-2}\eta^2\).

The normalised interaction shape is robust to the declared moderate constitutive perturbations, but the reciprocal amplitude prefactor is not transferable to those tensors.
