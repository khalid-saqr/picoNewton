# Step 9 architecture — robustness and claim lock

## Purpose

Step 9 freezes the Step 8 rank-one law and subjects it to constitutive and numerical perturbations without refitting its prefactor, vessel exponents or singular interaction mode. The outcome is a claim lock, not a new constitutive model.

## Frozen law

The only claim-bearing predictor is

\[
\widehat\Phi_1(\alpha,\eta,\mathbf g)
=C\alpha^{p_\alpha}\eta^{p_\eta}\Psi_1(\mathbf g),
\]

with the `reduced_law.json` and `step8_reduced_law.npz` artefacts consumed through checksum validation. Step 9 independently reconstructs the reciprocal operators only to verify continuity; reconstructed quantities never replace the frozen Step 8 parameters.

## General weak-anisotropy hierarchy

For

\[
\beta=b\varepsilon,\qquad \gamma=g\varepsilon,
\]

at fixed \(\delta\), the leading fields are

\[
U_\theta=\varepsilon g U_\theta^{(1)}+O(\varepsilon^3),
\qquad
U_z=U_z^{(0)}+\varepsilon^2bg U_z^{(2)}+O(\varepsilon^4).
\]

The signed-force kernel therefore separates as

\[
K^{(2)}(b,g,\delta)
=g^2K_{\gamma\gamma}(\delta)+bgK_{\beta\gamma}(\delta).
\]

This identity makes the beta-only path a strict null control at second order and allows nonreciprocal robustness to be evaluated without fitting a new amplitude law.

## Constitutive paths

The publication profile evaluates nine declared paths:

1. reciprocal \((b,g,\delta)=(1,1,1)\);
2. beta-low \((0.5,1,1)\);
3. gamma-low \((1,0.5,1)\);
4. gamma-only \((0,1,1)\);
5. beta-only \((1,0,1)\);
6. \((0.75,1.25,1)\);
7. \((1.25,0.75,1)\);
8. reciprocal with \(\delta=0.8\);
9. reciprocal with \(\delta=1.2\).

All non-null paths are tested across both Step 7 near-wall conditions, all six arteries and all 89 Step 8 waveforms.

## Two distinct robustness questions

Step 9 separates:

- **shape robustness:** whether the frozen rank-one interaction mode predicts the waveform dependence after using the exact path-specific kernel norm;
- **amplitude universality:** whether the unmodified reciprocal vessel prefactor predicts a nonreciprocal or diagonal-viscosity-shifted kernel.

Shape robustness may pass while amplitude universality fails. The latter outcome narrows the claim rather than invalidating the reciprocal law.

## Additional checks

- exact full-model closure at \(\varepsilon=0.08\);
- native endothelial-thickness perturbations from 0.8 to 1.2 times baseline;
- independent radial/quadrature pairs \((120,192)\) and \((180,384)\);
- backward residuals;
- beta-only null closure;
- checksum and source continuity with Step 8.

## Scope boundary

Step 9 does not fit a nonreciprocal constitutive prefactor, introduce biological endpoints, infer in-vivo thresholds or alter the Step 8 rank-one law. Its only extension is a robustness classification and final wording lock.
