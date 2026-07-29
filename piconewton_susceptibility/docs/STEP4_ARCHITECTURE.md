# Step 4 architecture: reciprocal weak-anisotropy hierarchy

## Purpose

Step 4 derives and verifies the reciprocal perturbation path

\[
\beta=\gamma=\varepsilon,\qquad \delta=1,
\]

without introducing the harmonic-interaction kernel, waveform susceptibility atlas, crossed artery-waveform experiment or critical-anisotropy inversion.

## Derived coefficient problems

For each pressure harmonic, the verified parent block equations are expanded as

\[
U_z=U_z^{(0)}+\varepsilon^2U_z^{(2)}+O(\varepsilon^4),
\qquad
U_\theta=\varepsilon U_\theta^{(1)}+O(\varepsilon^3).
\]

The implementation solves three linear boundary-value problems:

1. the isotropic axial field \(U_z^{(0)}\);
2. the first-order azimuthal field \(U_\theta^{(1)}\), forced by \(\mathcal L_0U_z^{(0)}\);
3. the second-order axial correction \(U_z^{(2)}\), forced by \(\mathcal L_1U_\theta^{(1)}\).

The second-order Lamb-vector excess coefficient is reconstructed in real time as

\[
\ell_r^{(2)}
=U_\theta^{(1)}\Omega_z^{(1)}
-U_z^{(2)}\Omega_\theta^{(0)}
-U_z^{(0)}\Omega_\theta^{(2)}.
\]

The signed force prediction is therefore

\[
\Delta F_s(t;\varepsilon)=\varepsilon^2F_s^{(2)}(t)+O(\varepsilon^4).
\]

## Publication experiment

All six arteries are evaluated at

\[
\varepsilon\in\{0.005,0.01,0.02,0.04,0.06,0.08,0.10\}.
\]

The order fit uses \(\varepsilon\le0.04\). Sign-reversed parity is tested at \(|\varepsilon|=0.08\).

## Mandatory gates

- a passing and checksum-valid Step 3 manifest;
- all six arteries;
- hierarchy and full-model residuals below \(10^{-10}\);
- even axial response, odd azimuthal response and even signed-force response under \(\varepsilon\mapsto-\varepsilon\);
- fitted orders within 0.98–1.02 for \(U_\theta\) and 1.95–2.05 for both the axial correction and signed-force excess;
- a contiguous 1% validity domain reaching at least \(\varepsilon=0.04\) for the two field coefficients and the signed-force waveform, RMS and peak metrics.

## Outputs

- `perturbation_coefficients.csv`;
- `epsilon_sweep.csv`;
- `order_slopes.csv`;
- `parity_checks.csv`;
- `validity_domains.csv`;
- `perturbation_waveforms.npz`;
- `step4_gate.json`;
- `step4_manifest.json`.

The exposure observable is reconstructed in time as a diagnostic. No exact bilinear kernel is claimed for the absolute-value operation.
