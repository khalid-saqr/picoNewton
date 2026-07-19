# Scientific model

## Objective

The model tests whether anisotropic near-wall forcing contains mechanosensory information that is not represented by wall shear stress alone.

The six arterial cases are:

- Aortic Root
- Thoracic Aorta
- Femoral
- Carotid
- Iliac
- Brachial

## Hydrodynamic observables

The Womersley solution generates an anisotropic and isotropic near-wall state for every artery. Three observables are retained independently.

### Wall shear stress

Wall shear stress is the tangential traction at the vessel wall:

\[
\tau_w(t)=\mu\left.\frac{\partial u_z}{\partial r}\right|_{r=R}.
\]

Units: Pa.

### Signed transverse Lamb force

The signed observable preserves the direction of the radial Lamb-vector contribution integrated over the near-wall endothelial control layer:

\[
F_{\mathrm{signed}}(t)=A_{EC}\int_{R-\delta_{EC}}^{R} f_r(r,t)\,dr.
\]

Units: N.

### Nonnegative force exposure

The magnitude-sensitive observable is

\[
F_{\mathrm{exposure}}(t)=A_{EC}\int_{R-\delta_{EC}}^{R}|f_r(r,t)|\,dr.
\]

Units: N.

It represents cumulative transverse forcing magnitude and is not interpreted as a signed normal load.

## Vector-resolved membrane-cortex mechanics

Tangential and normal tractions are passed through separate generalized standard-linear-solid models. Each direction contains fast and slow passive branches.

For a branch with modulus pair \(E_0,E_\infty\) and relaxation time \(\tau\), the complex compliance is of the standard-linear-solid form. The implementation evaluates periodic steady states in the frequency domain and verifies passivity over the configured domain.

The mechanical state preserves:

- tangential strain;
- signed normal strain;
- magnitude-sensitive normal exposure strain;
- apical and junctional tension;
- apical and junctional equivalent pressure;
- instantaneous force-vector angle.

The signed force and exposure pathways have independent localization areas and transfer fractions.

## Piezo1 coupling

Local membrane tension is converted to equivalent pressure through the declared curvature radius:

\[
P_{eq}(t)=\frac{2T(t)}{r_{eff}}.
\]

The pressure is supplied to the four-state Piezo1 model. Probability conservation and periodic closure are checked numerically.

The model returns:

\[
P_{Open}(t),
\]

\[
I(t)=N_{ch}\,g_{ch}\,P_{Open}(t)\,[V_m-E_{rev}],
\]

and a first-order calcium-scale endpoint derived from the calcium fraction of Piezo1 current, cell volume, Faraday constant, and clearance time.

The calcium endpoint is a reduced-order screening variable. It does not replace a full endothelial calcium-handling model.

## Comparison matrix

The workflow computes:

- zero-load controls;
- WSS-only response;
- absolute-WSS response;
- signed-force-only response;
- exposure-only response;
- combined vector response;
- RMS-matched force controls;
- peak-matched force controls;
- work-matched force controls;
- anisotropic versus isotropic responses;
- full harmonics versus harmonics 1-2;
- generalized viscoelastic versus elastic limits;
- leave-one-artery-out causal WSS surrogates;
- multifeature artery-response distances.

## Hypotheses

### H1: hydrodynamic nonredundancy

The anisotropic Lamb-force observables differ from WSS in waveform, phase, direction, or spectrum.

### H2: membrane-state nonredundancy

The corresponding membrane states remain distinct after passive mechanical filtering.

### H3: mechanosensory nonredundancy

H3a compares direct WSS and force pathways. H3b compares actual force pathways with leave-one-artery-out WSS-derived force surrogates.

### H4: anisotropy-specific response

The anisotropic response differs from the isotropic control under otherwise identical parameters.

### H5: higher-harmonic response

Harmonics above the second contribute a detectable endpoint difference.

### H6: artery specificity

Arteries are separated by a multifeature response vector rather than only by cycle mean.

### H7: interface robustness

The distinction persists under both viscoelastic and elastic mechanical limits.

## Interpretation boundary

The software calculates effect sizes and can apply explicit externally supplied thresholds. Thresholds, channel density, localization area, calcium conversion, and detection limits must remain traceable to independent evidence. The workflow does not modify parameters after observing outcomes.
