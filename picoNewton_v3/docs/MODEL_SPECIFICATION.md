# Mathematical model specification

## Preserved hydrodynamic model

The geometry is a straight, rigid, circular tube with axisymmetric, fully
developed velocity

\[
\mathbf u=(0,u_\theta(r,t),u_z(r,t)).
\]

For harmonic index \(h\), the dimensionless coupled system is

\[
i h\alpha^2 U_z=a_h+\mathcal L_0U_z+\beta\mathcal L_1U_\theta,
\]

\[
i h\alpha^2 U_\theta=\gamma\mathcal L_0U_z+\delta\mathcal L_1U_\theta,
\]

with

\[
\mathcal L_0=\frac{d^2}{dr^2}+\frac1r\frac d{dr},\qquad
\mathcal L_1=\mathcal L_0-\frac1{r^2}.
\]

Boundary conditions are axial symmetry and azimuthal regularity at the centre,
with no slip at the wall.

The vorticity components are

\[
\omega_\theta=-\partial_r u_z,\qquad
\omega_z=\frac1r\partial_r(ru_\theta).
\]

The radial Lamb-vector component is evaluated from reconstructed real fields:

\[
\ell_r=u_\theta\omega_z-u_z\omega_\theta.
\]

The signed and exposure force outputs are

\[
F_L^{\rm signed}=A_{\rm EC}\int_{R-\delta_{\rm EC}}^R\rho\ell_r\,dr,
\]

\[
F_L^{\rm exposure}=A_{\rm EC}\int_{R-\delta_{\rm EC}}^R|\rho\ell_r|\,dr.
\]

These are localized near-wall inertial control-volume quantities. They are not
identified with the exact total traction \(\boldsymbol\sigma\mathbf n\).

## Two-state mechanosensor

\[
C\underset{k_-}{\stackrel{k_+}{\rightleftharpoons}}O,
\qquad p(t)=\Pr(O),
\]

\[
\frac{dp}{dt}=k_+(\Psi)(1-p)-k_-(\Psi)p.
\]

The local-detailed-balance rates are

\[
k_+=\frac{p_0}{\tau_0}e^{\theta\Psi},\qquad
k_-=\frac{1-p_0}{\tau_0}e^{-(1-\theta)\Psi}.
\]

The Lamb-force work is

\[
\Psi_L=\frac{d_L\,\mathcal G_L(F_L)}{k_BT},
\]

where \(\mathcal G_L\) is signed, magnitude-sensitive, outward-only or
inward-only. The WSS control uses its own conjugate activation volume:

\[
\Psi_\tau=\frac{V_\tau\,\mathcal G_\tau(\tau_w)}{k_BT}.
\]

The minimal parallel-channel hypothesis is

\[
\Psi=\Psi_L+\Psi_\tau.
\]

For constant input,

\[
p_\infty=\operatorname{logistic}(\operatorname{logit}(p_0)+\Psi).
\]

The workflow computes the exact periodic fixed point of the affine one-cycle
state map; no arbitrary burn-in cycle count is required.

## Main dimensionless groups

\[
\Lambda_L=\frac{F_{L,\rm ref}d_L}{k_BT},\qquad
\Lambda_\tau=\frac{\tau_{w,\rm ref}V_\tau}{k_BT},
\]

\[
\Omega=\omega_0\tau_0,\qquad
\eta_0=\log\frac{p_0}{1-p_0}.
\]

## Claim boundary

The model can test observability, kinetic filtering, direction sensitivity and
nonredundancy relative to WSS controls. It cannot identify a receptor, prove
endothelial causality or equate the Lamb force with membrane tension.
