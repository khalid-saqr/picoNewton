# Step 1 Amendment 01A

## Scientific Reports successor: contract closure before Step 2

**Status:** Binding amendment to Step 1  
**Date:** 2026-07-29  
**Scope:** Scientific and numerical definitions only. The research objective, six-artery benchmark, mandatory result inventory, claim boundary, and remaining workflow are unchanged.

This addendum supersedes any conflicting Step 1 wording concerning the force observable, Fourier coefficients, dimensional scaling, crossed artery-waveform matrix, parent-solver authority, and mechanics verification.

---

## A1. Observable lineage and claim hierarchy

Let

\[
\boldsymbol{\ell}=\mathbf{u}\times\boldsymbol{\omega},
\qquad
\boldsymbol{\omega}=\nabla\times\mathbf{u},
\qquad
f_r(r,t)=\rho\,\ell_r(r,t).
\]

The published near-wall control-volume observable is retained exactly as the continuity and reproduction observable:

\[
F_{\mathrm{EC}}^{\mathrm{exposure}}(t)
=
A_{\mathrm{EC}}
\int_{R-\delta_{\mathrm{EC}}}^{R}
|f_r(r,t)|\,dr.
\]

The successor introduces the signed integral only as an analytical auxiliary:

\[
F_{\mathrm{EC}}^{\mathrm{signed}}(t)
=
A_{\mathrm{EC}}
\int_{R-\delta_{\mathrm{EC}}}^{R}
f_r(r,t)\,dr.
\]

The anisotropy-induced signed excess is

\[
\Delta F_{\mathrm{EC}}^{\mathrm{aniso}}(t;\boldsymbol{\epsilon})
=
F_{\mathrm{EC}}^{\mathrm{signed}}(t;\boldsymbol{\epsilon})
-
F_{\mathrm{EC}}^{\mathrm{signed}}(t;\mathbf{0}),
\]

where \(\boldsymbol{\epsilon}\) denotes the constitutive perturbation from the isotropic state \((\beta,\gamma,\delta)=(0,0,1)\).

The hierarchy is frozen as follows.

1. \(F_{\mathrm{EC}}^{\mathrm{exposure}}\) is the direct continuity observable inherited from the published paper.
2. \(F_{\mathrm{EC}}^{\mathrm{signed}}\) is the quantity used to derive the bilinear interaction kernel, cancellations, directional reversals, perturbative susceptibility, and critical anisotropy.
3. The manuscript must report both quantities. A conclusion supported only by the signed auxiliary must not be restated as a conclusion about the published magnitude-integrated exposure.
4. Neither quantity is a Cauchy wall traction. The terms **signed Lamb-force integral**, **magnitude-integrated Lamb exposure**, and **wall traction** must remain distinct.
5. No biological response, mechanotransduction threshold, disease prediction, or endothelial outcome is inferred from either hydrodynamic observable.

---

## A2. Canonical Fourier convention

### A2.1 Published one-sided convention

The parent reconstruction is

\[
G(t)
=
\Re\!\left[
\sum_{h=0}^{H}
\widehat G_h e^{ih\omega_0t}
\right].
\]

The same convention applies to each solved velocity and vorticity component.

### A2.2 Canonical two-sided convention for the successor

All kernel algebra uses

\[
G(t)=\sum_{h=-H}^{H}g_h e^{ih\omega_0t},
\qquad
g_{-h}=g_h^*.
\]

The exact conversion is

\[
g_0=\Re(\widehat G_0),
\qquad
g_h=\frac{\widehat G_h}{2},
\qquad
g_{-h}=\frac{\widehat G_h^*}{2},
\quad h=1,\ldots,H.
\]

The inverse conversion is \(\widehat G_h=2g_h\) for \(h>0\). No other factor-of-two convention is permitted.

### A2.3 Six published coefficients

The six tabulated coefficients are frozen as signed real coefficients for harmonics \(h=1,\ldots,6\):

\[
\widehat G_h=G_*a_h,
\qquad a_h\in\mathbb{R}.
\]

Hence

\[
g_{\pm h}=\frac{G_*a_h}{2}.
\]

A negative tabulated coefficient means a phase of \(\pi\) relative to a positive coefficient under this reconstruction; it is not an independently measured continuous phase. The six coefficients do not define a zero mode. Any \(h=0\) component must come from the frozen parent source and must never be inferred from the six-entry table.

Continuous phase rotations

\[
g_h\mapsto |g_h|e^{i\phi_h}
\]

are synthetic causal controls. They must be labelled **phase-rotation controls**, not physiological phase measurements.

Every implementation must round-trip a random conjugate-symmetric spectrum through both conventions and satisfy

\[
\frac{\|G_{\mathrm{one-sided}}-G_{\mathrm{two-sided}}\|_2}
{\max(\|G_{\mathrm{one-sided}}\|_2,\varepsilon_{\mathrm{mach}})}
\le 10^{-13}.
\]

---

## A3. Frozen dimensional scale and force benchmarks

The source hierarchy in A5 fixes the following dimensional constants.

| Quantity | Symbol | Frozen value | Unit |
|---|---:|---:|---:|
| Blood density | \(\rho\) | 1060 | kg m\(^{-3}\) |
| Axial kinematic viscosity | \(\nu_{zz}\) | \(3.5\times10^{-6}\) | m\(^2\) s\(^{-1}\) |
| Axial dynamic viscosity | \(\mu_{zz}=\rho\nu_{zz}\) | \(3.71\times10^{-3}\) | Pa s |
| Fundamental frequency | \(f_0\) | 1.2 | Hz |
| Fundamental angular frequency | \(\omega_0=2\pi f_0\) | \(2\pi\times1.2\) | rad s\(^{-1}\) |
| Pressure-gradient scale | \(G_*\) | \(1.0\times10^4\) | Pa m\(^{-1}\) |
| Endothelial reference area | \(A_{\mathrm{EC}}\) | \(100\times10^{-12}=1.0\times10^{-10}\) | m\(^2\) |
| Endothelial control volume | \(V_{\mathrm{EC}}\) | \(1.0\times10^{-15}\) | m\(^3\) |
| Near-wall integration depth | \(\delta_{\mathrm{EC}}=V_{\mathrm{EC}}/A_{\mathrm{EC}}\) | \(1.0\times10^{-5}\) | m |

The native dimensionless integration depth for artery \(i\) is

\[
\eta_i=\frac{\delta_{\mathrm{EC}}}{R_i}.
\]

The preregistered publication benchmark set is

\[
\mathcal F_*=\{1\ \mathrm{pN},\ 10\ \mathrm{pN}\},
\]

matching the reference contours used in the archived parent computation. The general critical-anisotropy API may accept any positive \(F_*\), but all primary manuscript comparisons must use \(\mathcal F_*\). Additional thresholds are sensitivity analyses and must be declared before the corresponding production run.

No dimensional constant or benchmark may be selected after inspection of successor results.

---

## A4. Corrected artery-waveform crossed design

Let

\[
\Phi_2=\Phi_2(\alpha,\eta,\mathbf g)
\]

be the second-order waveform susceptibility functional.

Two matrices are mandatory.

### A4.1 Hydrodynamic separation matrix

\[
\Phi_{ij}^{\mathrm{hydro}}
=
\Phi_2(\alpha_i,\eta_{\mathrm{ref}},\mathbf g_j),
\]

where

\[
\eta_{\mathrm{ref}}
=
\operatorname{median}_{i=1,\ldots,6}
\left(\frac{\delta_{\mathrm{EC}}}{R_i}\right)
=
2.361111\times10^{-3}.
\]

This matrix is the only crossed matrix permitted to support statements that isolate the Womersley-regime contribution from waveform structure.

### A4.2 Physiological native-depth matrix

\[
\Phi_{ij}^{\mathrm{phys}}
=
\Phi_2(\alpha_i,\eta_i,\mathbf g_j),
\qquad
\eta_i=\frac{\delta_{\mathrm{EC}}}{R_i}.
\]

This matrix preserves the physical integration depth and therefore combines the effects of \(\alpha_i\), radius-dependent \(\eta_i\), and waveform \(\mathbf g_j\). It must not be described as isolating only the Womersley effect.

The six native arteries remain the diagonal \(i=j\). The full \(6\times6\) cross-grid is mandatory for both matrices. The comparison must report at least:

- row, column, and interaction contributions;
- native-diagonal ranks;
- rank stability between \(\Phi^{\mathrm{hydro}}\) and \(\Phi^{\mathrm{phys}}\);
- whether the dominant harmonic pairs are preserved when \(\eta_i\) is restored.

---

## A5. Parent-model source of truth

The hydrodynamic source chain is frozen as:

1. **Publication:** K. M. Saqr, *A transverse picoNewton force revealed in anisotropic Womersley flow*, Scientific Reports 16, 12584 (2026), DOI `10.1038/s41598-026-47474-x`.
2. **Repository:** `khalid-saqr/picoNewton`.
3. **Published-source commit:** `4c3c36db0578373cc4e48d9d8c7e8a85944ed1cb`.
4. **Canonical parent artifact:** `picoNewton_v2.ipynb`.
5. **Canonical artifact blob:** `9d61c237cda75df338ce0383038f7765c886f503`.

Two execution modes are frozen.

### Historical reproduction mode

This mode reproduces the exact canonical parent artifact and is used only for regression, numerical lineage, and published-figure continuity. Its numerical idiosyncrasies are not automatically promoted into the successor derivation.

### Verified successor mode

This mode is a clean implementation of the published equations and boundary conditions, with:

- the same frozen physical inputs;
- Chebyshev-Gauss-Lobatto collocation and independently tested operator orientation;
- real-field reconstruction before every nonlinear velocity-vorticity multiplication;
- the canonical Fourier conversion in A2;
- direct access to signed, magnitude-integrated, isotropic, and anisotropy-excess observables;
- no mechanosensory, membrane, glycocalyx, cell, signalling, disease, or downstream biological modules.

Code from later `picoNewton_v3`, `picoNewton_v4`, or `LambForce-EC` mechanobiological workflows is outside the successor solver boundary. Reuse is permitted only for generic non-scientific utilities whose provenance is recorded and whose inclusion cannot alter equations, parameters, observables, or claims.

Step 2 must pin this source chain in a machine-readable registry before solver selection begins.

---

## A6. Mandatory Gromeka-Lamb mechanics closure

Under the frozen axisymmetric assumptions,

\[
\omega_\theta=-\partial_r u_z,
\qquad
\omega_z=\frac{1}{r}\partial_r(ru_\theta),
\]

and

\[
\ell_r
=u_\theta\omega_z-u_z\omega_\theta
=
\frac{\partial}{\partial r}
\left(\frac{u_z^2+u_\theta^2}{2}\right)
+
\frac{u_\theta^2}{r}.
\]

The following gates are mandatory before any susceptibility result is claim-bearing.

### Pointwise closure

For every artery, retained harmonic set, anisotropy state, and convergence grid,

\[
\mathcal E_{\mathrm{GL}}
=
\frac{
\|\ell_r^{(\mathbf u\times\boldsymbol\omega)}
-
\ell_r^{(\mathrm{identity})}\|_2
}
{
\max(\|\ell_r^{(\mathbf u\times\boldsymbol\omega)}\|_2,
\varepsilon_{\mathrm{mach}})
}
\le10^{-10}.
\]

### Integrated closure

The signed control-volume integral computed directly from \(\mathbf u\times\boldsymbol\omega\) and from the identity above must agree to relative error \(\le10^{-10}\).

### Isotropic boundary identity

For \(u_\theta=0\) and no slip at \(r=R\),

\[
\int_{R-\delta_{\mathrm{EC}}}^{R}\ell_r\,dr
=
-\frac{1}{2}u_z^2(R-\delta_{\mathrm{EC}},t).
\]

The verified solver must recover this identity to the same tolerance. This test explicitly exposes the signed integral's dependence on the inner control-volume boundary.

### Integration-depth sensitivity

Every primary result must be recomputed at

\[
\delta_{\mathrm{EC}}/2,
\qquad
\delta_{\mathrm{EC}},
\qquad
2\delta_{\mathrm{EC}}.
\]

A result may be called depth-robust only if its qualitative conclusion, dominant harmonic-pair ordering, susceptibility sign, and critical-anisotropy ordering are preserved at all three depths. Dimensional values must still be reported separately at each depth.

### Mechanics language

The manuscript and software documentation must preserve

\[
(\mathbf u\cdot\nabla)\mathbf u
=
\nabla\left(\frac{|\mathbf u|^2}{2}\right)
-
\boldsymbol\ell.
\]

Therefore \(\rho\boldsymbol\ell\) is one component of the convective-inertia decomposition. It is not the complete material acceleration and is not a Cauchy traction. Any comparison with wall force must calculate wall traction independently from the constitutive stress tensor.

---

## A7. Step 1 closure decision

The six stress-test deficiencies are closed by A1-A6. No change is made to:

- the successor paper's central question;
- the perturbative interaction-kernel derivation;
- the six native arteries;
- the full crossed artery-waveform design;
- the required harmonic-pair and phase analyses;
- the critical-anisotropy law;
- the comprehensive result inventory;
- the journal and claim boundary;
- the remaining workflow sequence.

**Step 1 status: CLOSED.**  
**Next authorised action: Step 2 only upon explicit instruction.**
