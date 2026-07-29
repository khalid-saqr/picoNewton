# Step 5 architecture: exact harmonic-interaction kernel

## Purpose

Step 5 converts the signed near-wall Lamb-force calculation into an exact quadratic input-output law. It consumes a passing Step 4 result and stops before defining the final waveform susceptibility functional or performing threshold inversion.

## Two kernels

### Exact full-model anisotropic-excess kernel

For the reciprocal state used in the parent continuity calculation,

\[
\beta=\gamma=\varepsilon,\qquad \delta=1,
\]

unit harmonic response fields define

\[
\widehat{\Delta F}_{s,q}
=\sum_{m+n=q}\Delta K_{mn}(\varepsilon)g_mg_n.
\]

The publication profile evaluates the exact kernel at \(\varepsilon=0.10\). The API accepts any admissible reciprocal value.

### Second-order kernel

The Step 4 hierarchy defines

\[
\widehat F^{(2)}_{s,q}
=\sum_{m+n=q}K^{(2)}_{mn}g_mg_n,
\]

where the radial integrand is

\[
U_{\theta,m}^{(1)}\Omega_{z,n}^{(1)}
-U_{z,m}^{(2)}\Omega_{\theta,n}^{(0)}
-U_{z,m}^{(0)}\Omega_{\theta,n}^{(2)}.
\]

This coefficient kernel is the derivation tool required by Step 6.

## Fourier convention

The one-sided parent coefficient \(\widehat G_h\) maps to

\[
g_h=\widehat G_h/2,\qquad g_{-h}=g_h^*.
\]

The output frequency is selected exactly by \(q=m+n\). Negative-frequency pairs therefore generate difference frequencies without a separate rule.

## Verification programme

Step 5 requires:

- native six-artery closure against direct real-field multiplication;
- deterministic synthetic-phase closure;
- Hermitian output symmetry and real waveform reconstruction;
- single-tone DC and frequency-doubling selection;
- two-tone DC, self-doubling, sum and difference frequencies;
- direct continuity with the Step 4 \(F_s^{(2)}(t)\) archive;
- comparison of \(\Delta K(0.10)/0.10^2\) with \(K^{(2)}\).

## Pair attribution

Raw ordered kernel entries are retained. For interpretation, \((m,n)\) and \((n,m)\) are combined into one unordered contribution. The reported absolute-share denominator is the sum of absolute pair contributions at fixed output frequency; it is not an energy fraction.

## Outputs

- `kernel_closure.csv`;
- `force_spectra.csv`;
- `kernel_entries.csv`;
- `pair_contributions.csv`;
- `dominant_pairs.csv`;
- `selection_rule_controls.csv`;
- `kernel_asymptotic_closure.csv`;
- `step4_kernel_continuity.csv`;
- `kernel_archive.npz`;
- `step5_gate.json`;
- `step5_manifest.json`.

## Boundary

The exact bilinear law applies only to the signed Lamb-force integral. The absolute-value exposure is not kernelised. Step 5 does not define susceptibility, cross artery-waveform cases, low-rank reduction or critical anisotropy.
