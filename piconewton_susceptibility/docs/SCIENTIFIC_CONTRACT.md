# Scientific Contract: Scientific Reports Successor

## Status

- **Project:** first successor to *A transverse picoNewton force revealed in anisotropic Womersley flow*
- **Target journal:** Scientific Reports
- **Parent article:** Khalid M. Saqr, *Scientific Reports* **16**, 12584 (2026)
- **DOI:** `10.1038/s41598-026-47474-x`
- **Working branch:** `successor/scirep-waveform-susceptibility`
- **Authoritative published notebook blob:** `picoNewton_v2.ipynb` at `9d61c237cda75df338ce0383038f7765c886f503`
- **Contract state:** locked for implementation; amendments require a dedicated contract revision commit before affected calculations are written

## 1. Successor-paper identity

The successor will be a perturbative and computational derivation of a **waveform susceptibility functional** that:

1. predicts the anisotropy-induced excess transverse near-wall Lamb-force response;
2. identifies the pressure-harmonic pairs and phase/sign relationships responsible for that response;
3. separates constitutive anisotropy, vessel Womersley response, waveform organisation, and dimensional force scale;
4. yields the critical constitutive anisotropy required to exceed any declared force benchmark;
5. demonstrates the predictive law using the six arterial configurations from the parent article.

The interaction kernel is a derivation and attribution tool. It is not the final scientific objective by itself.

## 2. Central research question

For an arbitrary periodic pressure-gradient waveform admitted by the published anisotropic Womersley model, what combination of Womersley regime, waveform harmonic content, harmonic phase/sign organisation, and constitutive anisotropy determines the anisotropy-induced excess transverse near-wall force, and what anisotropy is required to exceed a prescribed force level?

## 3. Protected parent model

The following features are inherited and will not be replaced in this successor:

- straight, rigid, circular vessel;
- axisymmetric, fully developed flow;
- velocity field

  \[
  \mathbf u(r,t)=u_\theta(r,t)\mathbf e_\theta+u_z(r,t)\mathbf e_z,
  \qquad u_r=0;
  \]

- periodic pressure-gradient forcing represented by discrete harmonics;
- the published linear anisotropic Newtonian constitutive tensor;
- dimensionless anisotropy ratios

  \[
  \beta=\nu_{z\theta}/\nu_{zz},\qquad
  \gamma=\nu_{\theta z}/\nu_{zz},\qquad
  \delta=\nu_{\theta\theta}/\nu_{zz};
  \]

- harmonic Womersley equations and their centreline regularity and wall no-slip conditions;
- vorticity definitions

  \[
  \omega_\theta=-\partial_r u_z,
  \qquad
  \omega_z=r^{-1}\partial_r(ru_\theta);
  \]

- radial Lamb-vector component

  \[
  \ell_r=u_\theta\omega_z-u_z\omega_\theta;
  \]

- the published endothelial-scale near-wall control-volume geometry;
- the isotropic limit \(\beta=\gamma=0,\ \delta=1\);
- the six arterial radii, Womersley numbers, and six-harmonic pressure-waveform coefficients listed below.

The historical notebooks remain unchanged. The successor must call an authoritative verified parent-model interface rather than fork the governing solver.

## 4. Primary constitutive path and robustness paths

### 4.1 Primary perturbation path

The derivation will use reciprocal weak anisotropy:

\[
\beta=\gamma=\varepsilon,\qquad \delta=1,
\qquad 0\leq\varepsilon\leq0.1.
\]

The expected hierarchy is

\[
U_{z,h}=U_{z,h}^{(0)}+\varepsilon^2U_{z,h}^{(2)}+O(\varepsilon^4),
\]

\[
U_{\theta,h}=\varepsilon U_{\theta,h}^{(1)}+O(\varepsilon^3).
\]

### 4.2 Secondary robustness paths

The main theorem will not be silently generalised beyond the reciprocal path. Robustness tests will use a small declared set of:

- nonreciprocal cases with \(\beta\neq\gamma\) while \(|\beta|,|\gamma|\leq0.1\);
- diagonal variations \(\delta\in\{0.9,1.0,1.1\}\);
- sign-reversed reciprocal coupling where mathematically admissible.

These tests assess robustness; they do not create an unrestricted four-dimensional constitutive sweep.

## 5. Force observables

### 5.1 Primary signed quantity

The theoretical response is the signed control-volume Lamb-force proxy

\[
F_s(t)=A_{\mathrm{EC}}
\int_{R-\delta_{\mathrm{EC}}}^{R}
\rho\ell_r(r,t)\,dr.
\]

The anisotropy-induced excess is

\[
\Delta F_s(t;\varepsilon)
=F_s(t;\varepsilon)-F_s(t;0).
\]

This isotropic subtraction is mandatory.

### 5.2 Secondary exposure quantity

The nonnegative exposure quantity is

\[
F_{\mathrm{exp}}(t)=A_{\mathrm{EC}}
\int_{R-\delta_{\mathrm{EC}}}^{R}
|\rho\ell_r(r,t)|\,dr.
\]

It is reconstructed in the time domain and reported separately. The exact bilinear interaction kernel is not claimed for the absolute-value exposure operation.

### 5.3 Required response measures

The package will calculate at least:

- cycle RMS excess force;
- peak absolute excess force;
- harmonic-resolved signed excess force;
- inward and outward duty fractions;
- high-harmonic force fraction;
- isotropic-normalised excess;
- wall-shear-force-normalised excess as a scale comparison only.

The Lamb-force quantity is not Cauchy traction and is not the complete radial material acceleration.

## 6. Exact harmonic interaction law

Use the two-sided representation

\[
G(t)=G_*\sum_{h=-H}^{H}g_h e^{ih\omega_0t},
\qquad g_{-h}=g_h^*.
\]

For each harmonic, the published linear flow problem defines a response operator

\[
\mathbf U_h=\mathcal H_h(\alpha,\beta,\gamma,\delta)g_h.
\]

The signed anisotropic-excess force spectrum must be represented and verified as

\[
\widehat{\Delta F}_{s,q}
=\rho A_{\mathrm{EC}}U_*^2
\sum_{m+n=q}
\Delta\mathcal K_{mn}^{(q)}
(\alpha,\beta,\gamma,\delta,\eta)g_mg_n,
\]

where

\[
\eta=\delta_{\mathrm{EC}}/R.
\]

The kernel must retain DC, sum-frequency, difference-frequency, and frequency-doubling contributions through the negative-frequency conjugate representation.

Terminology is restricted to **quadratic frequency mixing**, **harmonic interaction**, **mode-pair coupling**, and **spectral generation**. A nonlinear kinetic-energy cascade is not claimed.

## 7. Waveform susceptibility functional

For the reciprocal weak-anisotropy path, the primary target law is

\[
\Delta F_{\mathrm{rms}}
=\rho A_{\mathrm{EC}}U_*^2
\varepsilon^2
\Phi_{2,\mathrm{rms}}(\alpha,\eta,\mathbf g)
+O(\varepsilon^4).
\]

Companion functionals will be defined for peak, harmonic-resolved, and directional response:

\[
\Phi_{2,\mathrm{peak}},\qquad
\Phi_{2,q},\qquad
\Phi_{2,+},\qquad
\Phi_{2,-}.
\]

The law must preserve complex harmonic information. Pressure RMS or harmonic magnitudes alone are not assumed sufficient.

## 8. Critical-anisotropy prediction

For a declared force benchmark \(F_*\), the perturbative estimate is

\[
\varepsilon_{\mathrm{crit}}^{(2)}
=\left[
\frac{F_*}
{\rho A_{\mathrm{EC}}U_*^2\Phi_{2,\mathrm{rms}}}
\right]^{1/2}.
\]

Every reported critical value must include:

- the perturbative estimate;
- full-model refinement or confirmation;
- perturbative-domain validity status;
- threshold-reachability status;
- the exact force metric to which \(F_*\) applies.

A declared force benchmark is not called an endothelial activation threshold unless independent experimental evidence is supplied. The software must return an explicit unreachable or out-of-domain state rather than extrapolate silently.

## 9. Frozen six-artery input inventory

The following parent-paper inputs are mandatory and immutable unless a source-correction contract is approved.

| Artery/segment | Radius \(R\) (m) | \(\alpha\) | Six published waveform coefficients |
|---|---:|---:|---|
| Aortic Root | 0.0150 | 22.03 | `[1.00, 0.82, 0.54, 0.33, 0.24, 0.17]` |
| Thoracic Aorta | 0.0120 | 17.62 | `[1.00, 0.76, 0.45, 0.28, 0.20, 0.12]` |
| Femoral | 0.0040 | 5.87 | `[1.00, 0.58, 0.10, -0.17, 0.05, 0.04]` |
| Carotid | 0.0035 | 5.14 | `[1.00, 0.63, 0.31, 0.15, 0.10, 0.06]` |
| Iliac | 0.0045 | 6.61 | `[1.00, 0.51, 0.12, -0.11, 0.05, 0.03]` |
| Brachial | 0.0020 | 2.94 | `[1.00, 0.49, 0.16, -0.05, 0.02, 0.01]` |

The coefficients are preserved as the published signed waveform representation. Synthetic continuous phase rotations are treated as controls and are not attributed to the physiological data source.

## 10. Six-artery experimental design

The arteries are the physiological backbone of the paper and enter at four levels.

### 10.1 Native reproduction

Recover every published configuration

\[
(\alpha_i,\mathbf g_i),\qquad i=1,\ldots,6.
\]

### 10.2 Dimensionless native susceptibility

Calculate \(\Phi_2(\alpha_i,\mathbf g_i)\) after separating the dimensional force scale.

### 10.3 Crossed vessel-waveform experiment

Calculate the full matrix

\[
\Phi_{ij}=\Phi_2(\alpha_i,\mathbf g_j),
\qquad i,j=1,\ldots,6.
\]

All 36 entries are mandatory. The diagonal recovers the native cases. The off-diagonal entries separate Womersley response from waveform organisation.

### 10.4 Harmonic and phase/sign ablations

For each artery:

- remove each harmonic individually;
- retain RMS while redistributing harmonic content in declared synthetic controls;
- compare native signed coefficients with sign-neutralised controls;
- apply controlled phase alignment and phase scrambling only to synthetic extensions;
- identify constructive and destructive harmonic-pair contributions.

## 11. Mandatory result groups

The publication workflow must generate all eight result groups below.

### R1. Parent-model continuity

- isotropic analytical verification;
- selected historical notebook regression;
- all six native arteries;
- signed force, exposure, and isotropic excess;
- radial, temporal, and near-wall quadrature convergence.

### R2. Perturbative hierarchy and validity domain

- \(U_\theta=O(\varepsilon)\);
- \(\Delta F=O(\varepsilon^2)\) on the primary path;
- full-versus-perturbative error maps;
- validated \(\varepsilon\)-domain for each artery and response metric.

### R3. Exact harmonic selection and kernel verification

- DC generation;
- frequency doubling;
- sum and difference frequencies;
- conjugate symmetry;
- agreement between kernel and direct time-domain reconstruction.

### R4. Harmonic-pair attribution

- contribution matrices \(C_{mn\rightarrow q}\);
- constructive and destructive interference;
- dominant pairs for every artery and major output harmonic.

### R5. Waveform organisation controls

- equal-RMS comparisons;
- harmonic-removal controls;
- spectral-slope controls;
- sign and synthetic-phase controls;
- quantitative determination of whether RMS, bandwidth, or phase/sign organisation governs susceptibility.

### R6. Six-artery susceptibility atlas

For every artery report:

- RMS and peak susceptibility;
- harmonic-resolved susceptibility;
- dominant mode pairs;
- inward/outward duty fractions;
- high-harmonic fraction;
- perturbative validity range;
- critical-anisotropy results for the declared force benchmarks.

### R7. Crossed six-by-six susceptibility matrix

Report:

- the complete \(6\times6\) matrix;
- native diagonal ranking;
- vessel-effect component;
- waveform-effect component;
- vessel-waveform interaction component.

### R8. Critical-anisotropy prediction

For every valid artery, waveform, and declared benchmark:

- perturbative estimate;
- full-model crossing;
- relative prediction error;
- unreachable and out-of-domain cases.

## 12. Conditional result groups

### C1. Low-rank kernel reduction

A one-to-three-mode reduced representation may be promoted only if it meets the held-out validation gate in Section 14.

### C2. Compact scalar waveform index

A descriptor based on weighted harmonic energy, spectral centroid, phase/sign coherence, or related quantities may be promoted only if it reproduces the exact susceptibility with the declared held-out accuracy. Otherwise it remains a negative result.

Failure of C1 or C2 does not invalidate R1-R8.

## 13. Computational experiment families

The package will include only the following experiment families:

1. single-tone analytical controls;
2. two-tone analytical controls;
3. sparse three-tone controls;
4. controlled spectral-slope families;
5. controlled phase-alignment and phase-scrambling families;
6. harmonic-removal and sign controls based on the six arteries;
7. six native physiological configurations;
8. the crossed \(6\times6\) vessel-waveform matrix;
9. limited nonreciprocal and diagonal-viscosity robustness cases.

A broad random waveform or biological-parameter Sobol programme is outside this successor.

## 14. Predeclared numerical and scientific gates

### G1. Parent-model verification

- normalised linear-system residual: \(\leq10^{-10}\);
- isotropic analytical relative \(L_2\) error: \(\leq10^{-9}\);
- selected parent-workflow regression error: \(\leq10^{-6}\), subject to the authoritative solver-mode convention;
- primary force metrics stable within 1% under the publication-resolution convergence check.

### G2. Kernel equivalence

For signed force:

- direct-versus-kernel waveform relative \(L_2\) error: \(\leq10^{-8}\);
- direct-versus-kernel spectral relative error: \(\leq10^{-8}\);
- power outside the analytically supported output band must be at numerical-noise level.

### G3. Perturbative scaling

On the declared weak-anisotropy interval:

- fitted log-log slope of excess RMS force versus \(|\varepsilon|\): between 1.9 and 2.1;
- perturbative relative error \(\leq5\%\) for \(|\varepsilon|\leq0.05\);
- perturbative relative error \(\leq10\%\) at \(|\varepsilon|=0.1\), otherwise the validated domain is reduced and reported.

No artery is forced to pass the same maximal \(\varepsilon\); artery-specific validity limits are allowed.

### G4. Critical-anisotropy inversion

When a monotone crossing exists inside the validated range:

- full-model force at the predicted threshold must agree with the requested benchmark within 5%;
- refined full-model \(\varepsilon_{\mathrm{crit}}\) must differ from the perturbative estimate by no more than 10% for the perturbative formula to be called predictive.

Otherwise the case is labelled refined-only, unreachable, nonmonotone, or out-of-domain.

### G5. Six-artery completion

- all six native cases complete;
- all 36 crossed cases complete;
- every case has provenance, convergence status, and source-data export;
- no failed case is silently omitted from rankings or figures.

### G6. Harmonic attribution closure

The sum of pairwise contributions must reconstruct each reported signed output harmonic within \(10^{-8}\) relative error, with an absolute tolerance used near zero.

### G7. Conditional low-rank promotion

C1 may be promoted only if one to three modes achieve:

- \(\leq10\%\) relative error on held-out waveform cases;
- \(\leq10\%\) relative error under leave-one-artery-out validation;
- preservation of the native six-artery susceptibility ranking except where uncertainty intervals overlap.

### G8. Conditional scalar-index promotion

C2 may be promoted only if the frozen descriptor achieves:

- \(R^2\geq0.90\) on development cases;
- \(\leq15\%\) median relative error on held-out cases;
- no post hoc alteration after held-out evaluation.

### G9. Outcome neutrality

Null or negative results for phase sensitivity, low-rank reduction, compact indexing, nonreciprocal robustness, or threshold reachability must be retained. Thresholds and gates are not changed after observing publication-profile results.

## 15. Permitted claims

Subject to the gates, the paper may claim:

- an exact harmonic interaction law for signed near-wall Lamb-force response within the published anisotropic Womersley model;
- a verified weak-anisotropy susceptibility law on its measured validity domain;
- mode-pair and phase/sign attribution of waveform susceptibility;
- a predictive critical-anisotropy relation for declared force benchmarks;
- separation of waveform and Womersley-regime effects through the crossed six-artery design;
- waveform-level generality for arbitrary periodic inputs admitted by the model.

## 16. Prohibited claims

The paper and software must not claim:

- a universal endothelial activation threshold;
- patient-specific disease prediction;
- population prevalence or clinical risk from the six arteries;
- universality across arbitrary geometries, compliant walls, or constitutive laws;
- exact wall traction, membrane tension, or complete radial acceleration;
- a receptor-specific or Piezo1 mechanism;
- a nonlinear kinetic-energy cascade;
- experimental validation of the anisotropy coefficients;
- in-vivo sufficiency of the straight-tube model.

The word **significant** must always be tied to a declared mathematical benchmark, comparison scale, uncertainty test, or statistical test.

## 17. Robustness inventory

The supplementary calculation set must include:

- radial resolution;
- time resolution;
- near-wall quadrature resolution;
- harmonic truncation;
- control-volume thickness;
- selected \(\delta\)-variation;
- selected nonreciprocal \((\beta,\gamma)\) cases;
- force-benchmark sensitivity;
- signed-versus-exposure comparison;
- perturbative-domain sensitivity.

## 18. Publication output inventory

The final workflow must export:

- exact configuration files;
- parent-source and Git provenance;
- six-artery frozen input table;
- harmonic response fields;
- interaction kernels;
- pair-contribution tables;
- susceptibility tables;
- crossed \(6\times6\) matrix;
- critical-anisotropy tables;
- convergence and gate tables;
- figure-source CSV files;
- array data in HDF5 or compressed NumPy format;
- environment manifest;
- checksums;
- claim-retention report.

## 19. Main-paper result architecture

The intended six main figures are:

1. parent-model continuity and perturbative hierarchy;
2. exact interaction law and kernel equivalence;
3. waveform amplitude, sign, and phase organisation controls;
4. susceptibility landscape and perturbative validity;
5. six-artery physiological susceptibility atlas;
6. crossed matrix and critical-anisotropy prediction.

Robustness, nonreciprocity, diagonal-viscosity variation, and extended convergence belong in Supplementary Information unless they overturn a principal claim.

## 20. Software boundary

The new implementation will live under

```text
piconewton_susceptibility/
```

It will:

- depend on an authoritative verified parent-model interface;
- reuse existing Colab, Google Drive, checkpoint, manifest, and checksum infrastructure where technically compatible;
- contain no mechanosensor, ion-channel, membrane, or calcium-current model;
- avoid modifying `picoNewton_v1.ipynb`, `picoNewton_v1.py`, or `picoNewton_v2.ipynb`;
- keep the scientific implementation in tested Python modules and use the notebook for orchestration and presentation.

## 21. Step-1 completion criteria

Step 1 is complete when:

- this contract is committed on the working branch;
- the parent article, published notebook blob, six artery inputs, observables, mandatory results, conditional results, gates, claims, and non-claims are frozen;
- no implementation code has yet been introduced;
- the next step cannot alter this scope without a visible contract revision.
