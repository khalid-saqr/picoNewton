# Step 6 — standard-linear-solid membrane–cortex interface

## Scope

Step 6 derives and validates a passive reduced-order standard-linear-solid (SLS) interface. Wall shear stress and the signed near-wall control-volume force are mapped separately to a common scalar equivalent stress and then propagated to the common membrane state, **equivalent areal strain**. No Piezo1 gating is calculated in this step.

## Constitutive model

The SLS consists of a relaxed spring `E_inf` in parallel with a Maxwell branch containing spring `E_1 = E_0 - E_inf` and dashpot `eta = E_1 tau_sigma`.

For harmonic convention `exp(i omega t)`, the creep compliance is

```text
J*(omega) = (1 + i omega tau_sigma) /
            (E_inf + i omega tau_sigma E_0)
```

This gives the correct limits:

```text
J*(0)        = 1 / E_inf
J*(infinity) = 1 / E_0
```

The elastic limit is obtained by setting `E_0 = E_inf`, which gives zero loss modulus and zero dashpot dissipation.

## Reference parameters

| Parameter | Value |
|---|---:|
| Instantaneous modulus `E_0` | 2500 Pa |
| Relaxed modulus `E_inf` | 1000 Pa |
| Stress-relaxation time `tau_sigma` | 0.25 s |
| Cortex thickness | 0.35 µm |
| Maxwell modulus | 1500 Pa |
| Derived viscosity | 375 Pa s |
| Derived creep time | 0.625 s |

These values lie inside the source-locked engineering envelope. They are reference values, not fitted endothelial parameters.

## Separate load mappings

### WSS pathway

```text
sigma_eq^WSS(t) = chi_WSS tau_w(t)
```

The reference transfer fraction is `chi_WSS = 1`. WSS remains a signed tangential traction.

### Force pathway

```text
sigma_eq^F(t) = chi_F F_signed(t) / A_eff
```

The reference values are `chi_F = 1` and `A_eff = 100 µm²`. The signed force is required. The nonnegative magnitude exposure is not used as signed mechanical loading.

The two pathways are not added or rotated into one another in Step 6. They are processed independently through the same SLS operator so they can be compared at a common membrane-state level.

## Passivity and energy

The loss modulus is nonnegative over the tested frequency range. For a sinusoidal stress, mean mechanical input power equals dashpot dissipation to numerical precision. Instantaneous dissipation density is `q² / eta`, where `q` is Maxwell-branch stress, and is therefore nonnegative for all admissible parameters.

## Publication-resolution reference run

| Artery | Pathway | RMS equivalent stress (Pa) | RMS strain | Peak absolute strain |
|---|---|---:|---:|---:|
| aortic_root | force_signed_anisotropic | 0.176537 | 0.000119768 | 0.000302963 |
| aortic_root | wss_anisotropic | 5.18817 | 0.00224294 | 0.00478054 |
| brachial | force_signed_anisotropic | 0.0189639 | 1.48052e-05 | 2.76322e-05 |
| brachial | wss_anisotropic | 1.88445 | 0.000828006 | 0.00144298 |
| carotid | force_signed_anisotropic | 0.0624968 | 4.56451e-05 | 9.92781e-05 |
| carotid | wss_anisotropic | 3.26484 | 0.00142553 | 0.00270615 |
| femoral | force_signed_anisotropic | 0.0468861 | 3.65177e-05 | 6.58272e-05 |
| femoral | wss_anisotropic | 2.96227 | 0.00129772 | 0.00223206 |
| iliac | force_signed_anisotropic | 0.0379618 | 2.98467e-05 | 5.35031e-05 |
| iliac | wss_anisotropic | 2.68359 | 0.00117831 | 0.00198728 |
| thoracic_aorta | force_signed_anisotropic | 0.0954157 | 6.64638e-05 | 0.000160102 |
| thoracic_aorta | wss_anisotropic | 3.89419 | 0.0016895 | 0.00345204 |

The reference mapping predicts substantially smaller force-derived equivalent stress than WSS-derived stress because the signed picoNewton force is divided by the fixed 100 µm² reference area. This is a result of the declared reference mapping, not evidence that WSS is biologically dominant.

## Validation gates

- zero input gives zero strain;
- stress-step strain begins at `sigma/E_0` and relaxes to `sigma/E_inf`;
- low- and high-frequency limits match the analytical compliance;
- the elastic limit matches Hookean response;
- branch tensions close to the applied resultant;
- loss modulus and dissipation are nonnegative;
- harmonic input power equals dashpot dissipation;
- all six arterial waveforms produce finite outputs;
- the signed force, isotropic force and anisotropy increment remain separate;
- magnitude exposure is never used as signed loading;
- Piezo1 coupling is not executed.

## Claim boundary

This step validates the mathematics, units, passivity and numerical implementation of a reduced-order SLS interface. It does not establish that the reference transfer fractions or effective area are unique endothelial values. Those quantities remain explicit uncertainty parameters for the later six-artery hypothesis analysis.
