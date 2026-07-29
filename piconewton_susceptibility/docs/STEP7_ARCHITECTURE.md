# Step 7 architecture: crossed susceptibility and waveform experiments

## Purpose

Step 7 separates vessel response, near-wall geometry and waveform organisation. It consumes a passing Step 6 result and executes the complete predeclared waveform experiment programme without fitting the reduced law reserved for Step 8.

## Crossed matrices

Two complete six-by-six matrices are mandatory.

### Hydrodynamic matrix

\[
\Phi^{\rm hydro}_{ij}=\Phi_{2,\rm rms}(\alpha_i,\eta_{\rm ref},\mathbf g_j),
\qquad \eta_{\rm ref}=2.361111\times10^{-3}.
\]

Holding \(\eta\) fixed isolates the Womersley response regime from waveform organisation.

### Physiological matrix

\[
\Phi^{\rm phys}_{ij}=\Phi_{2,\rm rms}(\alpha_i,\eta_i,\mathbf g_j),
\qquad \eta_i=\delta_{\rm EC}/R_i.
\]

This matrix preserves the native near-wall thickness ratio. In both matrices, the diagonal contains the six native artery-waveform pairs and the off-diagonal entries are counterfactual waveform transfers, not additional physiological measurements.

## Exact validation

Every one of the 72 matrix entries is evaluated with both the Step 5 second-order kernel and the exact reciprocal excess kernel at \(\varepsilon=0.08\). Waveform, RMS and spectrum errors must remain below 1%.

## Waveform controls

For each native artery, Step 7 calculates:

- removal of each of the six harmonics;
- the same six removals after restoring the original input RMS;
- sign neutralisation;
- zero-phase alignment and a common coherent phase;
- eight deterministic phase scrambles preserving harmonic magnitudes.

For the real signed native coefficients, sign neutralisation and zero-phase alignment are algebraically identical. Their equality is audited and not counted as independent evidence.

## Causal analytical families

All synthetic families have unit pressure-waveform RMS:

- six single tones;
- fifteen two-tone combinations with a fixed relative phase;
- three sparse three-tone controls;
- five spectral-slope controls.

These controls diagnose harmonic content and phase organisation. They are not attributed to measured arterial waveforms.

## Decomposition

The raw and logarithmic matrices are decomposed into vessel, waveform and interaction sums of squares. The logarithmic decomposition is diagnostic evidence for possible multiplicative separability; Step 7 does not fit or promote a reduced law.

## Outputs

- `crossed_susceptibility.csv`;
- `crossed_exact_validation.csv`;
- `crossed_variance_decomposition.csv`;
- `crossed_main_effects.csv`;
- `step6_native_continuity.csv`;
- `native_waveform_controls.csv`;
- `control_degeneracy_audit.csv`;
- `causal_waveform_families.csv`;
- `response_residuals.csv`;
- `step7_archive.npz`;
- `step7_gate.json`;
- `step7_manifest.json`.

## Boundary

Step 7 performs no low-rank fitting, compact-index selection, leave-one-artery-out reduced-law validation, nonreciprocal sweep, diagonal-viscosity variation, exposure kernel or mechanosensory calculation.
