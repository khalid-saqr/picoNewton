# Step 8 architecture — reduced predictive law

## Scientific purpose

Step 8 converts the complete Step 7 experiment dataset into a predictive reduction. It tests whether vessel dependence can be represented by a power law in Womersley number and near-wall thickness ratio, and whether the two-sided interaction kernel can be replaced by one, two or three singular modes without losing waveform generality.

## Reduced law

For each Step 7 vessel and near-wall condition, let \(K(\alpha,\eta)\in\mathbb C^{12\times12}\) be the dimensionless second-order interaction kernel and let

\[
S(\alpha,\eta)=\|K(\alpha,\eta)\|_F.
\]

The vessel scale is fitted as

\[
S(\alpha,\eta)=C\alpha^{p_\alpha}\eta^{p_\eta}.
\]

The normalised universal kernel is

\[
\overline K=\frac{1}{N}\sum_{i=1}^{N}\frac{K_i}{\|K_i\|_F}.
\]

Its rank-\(R\) approximation is

\[
\overline K_R=\sum_{r=1}^{R}\sigma_r u_rv_r^{\dagger},\qquad R\in\{1,2,3\}.
\]

For a real pressure waveform with one-sided coefficients \(g_h\), the canonical two-sided coefficients are \(c_{\pm h}=g_{\pm h}/2\). The output spectrum is

\[
\widehat\phi_q^{(R)}=\sum_{m+n=q}(\overline K_R)_{mn}c_mc_n,
\]

and the phase-aware waveform functional is

\[
\Psi_R(\mathbf g)=\left(\sum_{q=-12}^{12}|\widehat\phi_q^{(R)}|^2\right)^{1/2}.
\]

The predictive law is therefore

\[
\widehat\Phi_R(\alpha,\eta,\mathbf g)
=C\alpha^{p_\alpha}\eta^{p_\eta}\Psi_R(\mathbf g).
\]

## Validation design

Each of the six arteries is held out in turn. Both its hydrodynamic and physiological kernels are excluded. The remaining ten kernels determine the vessel power law and universal kernel. The held-out artery is then predicted for 89 waveforms:

- six native arterial waveforms;
- six equal-RMS single tones;
- fifteen equal-RMS two-tone controls;
- three sparse three-tone controls;
- five spectral-slope controls;
- fifty-four deterministic phase challenges.

This produces 1,068 predictions per candidate rank. Validation thresholds are frozen before model selection: median error at most 5%, 90th-percentile error at most 12%, maximum error at most 20%, each waveform-family median at most 5%, each family maximum at most 20%, and rank-order Spearman correlation at least 0.95.

## Negative comparator

A simpler scalar moment is tested under simultaneous artery and waveform-family holdout:

\[
\widehat\Phi_{\mathrm{scalar}}
=C\alpha^{p_\alpha}\eta^{p_\eta}
\sum_{h=1}^{6}h^{-s}|g_h|^2.
\]

It is selected only if it meets the same family-general thresholds. Failure is recorded rather than repaired post hoc.

## Scope boundary

Step 8 uses only the reciprocal \(\beta=\gamma=\varepsilon\), \(\delta=1\) second-order kernel, the straight rigid tube model, the Step 7 six-harmonic waveform class and the tested \((\alpha,\eta)\) domain. Constitutive robustness and final claim locking remain Step 9.
