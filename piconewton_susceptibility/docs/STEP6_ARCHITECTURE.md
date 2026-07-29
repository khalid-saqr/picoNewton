# Step 6 architecture: waveform susceptibility and critical anisotropy

## Purpose

Step 6 converts the verified second-order signed-force kernel into dimensionless waveform-susceptibility functionals and implements a fail-closed inverse predictor for reciprocal constitutive anisotropy. It consumes passing Step 4 and Step 5 artefacts and stops before crossed vessel-waveform experiments, waveform ablations, low-rank reduction and constitutive robustness.

## Primary functional

For each native artery,

\[
\Delta F_s(t;\varepsilon)
=\rho A_{\rm EC}U_*^2\varepsilon^2\Phi_2(t)+O(\varepsilon^4),
\qquad
U_*=G_*R^2/\mu_{zz}.
\]

The primary scalar is

\[
\Phi_{2,\rm rms}=\operatorname{rms}[\Phi_2(t)].
\]

Secondary quantities are peak absolute susceptibility, positive-part RMS, negative-part RMS, mean, inward/outward duty fractions and the complex harmonic coefficients \(\Phi_{2,q}\). The positive and negative RMS values are directional decompositions; they are not separate forces.

## Scale separation

The dimensional scale is

\[
S_F=\rho A_{\rm EC}U_*^2.
\]

The implementation must verify

\[
F_s^{(2)}(t)=S_F\Phi_2(t)
\]

and invariance of \(\Phi_2\) when the pressure-gradient scale is changed while waveform shape, \(\alpha\) and \(\eta\) are held fixed.

## Harmonic functional

The Step 5 spectrum is normalised as

\[
\Phi_{2,q}=\widehat F^{(2)}_{s,q}/S_F,
\]

with all frequencies \(q=-12,\ldots,12\) retained. Parseval closure between the harmonic and time-domain RMS is mandatory.

## Critical anisotropy

For metric \(M\in\{\mathrm{rms},\mathrm{peak}\}\),

\[
\varepsilon_{\rm crit}^{(2)}=\sqrt{F_*/F_M^{(2)}}.
\]

The preregistered manuscript benchmarks remain 1 pN and 10 pN. RMS is the primary inversion metric and peak is secondary. Every row must report the perturbative estimate, artery-specific Step 4 validity limit, exact full-model value at that limit, reachability, and a refined full-model crossing when one exists. Estimates outside the validated domain are retained as formal second-order estimates but are never promoted as validated predictions. Estimates with \(\varepsilon\ge1\) are additionally marked constitutively inadmissible because reciprocal positive definiteness with \(\delta=1\) requires \(|\varepsilon|<1\).

## Full-model validation

Exact reciprocal solutions are evaluated at \(\varepsilon=0.04,0.08,0.10\). The mandatory accuracy gate applies through \(\varepsilon=0.08\), the common strict Step 4 domain. Inversion is independently verified using exact targets generated at \(\varepsilon=0.04\) and \(0.08\).

## Outputs

- `native_susceptibility.csv`;
- `harmonic_susceptibility.csv`;
- `exact_susceptibility_validation.csv`;
- `pressure_scale_invariance.csv`;
- `inverse_verification.csv`;
- `critical_anisotropy.csv`;
- `step5_susceptibility_continuity.csv`;
- `susceptibility_archive.npz`;
- `step6_gate.json`;
- `step6_manifest.json`.

## Boundary

Step 6 does not cross the six vessel regimes with the six waveforms, remove or rotate harmonics, derive a low-rank predictor, vary \(\beta\), \(\gamma\) or \(\delta\), construct an exposure kernel, or interpret the force benchmarks as biological activation thresholds.
