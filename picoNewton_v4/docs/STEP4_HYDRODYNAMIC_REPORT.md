# Step 4 — standalone hydrodynamic reproduction report

**Profile:** publication  
**Resolution:** radial order 150; 2,048 time points; 256 near-wall quadrature nodes  
**Arteries:** six fixed ground-truth cases  
**Membrane/Piezo1 coupling:** absent by design

## Implementation

The v4 hydrodynamic layer independently implements the verified Chebyshev collocation formulation. It reconstructs real velocity and vorticity fields before evaluating the nonlinear Lamb-vector product. The package retains distinct arrays for WSS, signed force, magnitude exposure, isotropic reference and anisotropy-specific increments.

A process-wide single-thread BLAS limit is applied. This is necessary for predictable execution in Colab and avoids severe oversubscription stalls while retaining deterministic dense linear solves for the modest spectral systems.

## Publication-profile validation

| Test | Observed | Required | Result |
|---|---:|---:|:---:|
| Maximum paper-table Womersley rounding difference | 0.0140316 | < 0.02 | Pass |
| Maximum isotropic analytical L∞ error | 1.08026e-13 | < 1e-8 | Pass |
| Maximum normalized backward residual | 3.58783e-17 | < 1e-13 | Pass |
| Maximum Gromeka–Lamb closure error | 2.28817e-13 | < 1e-10 | Pass |
| Signed-force power above h=12 | 3.54734e-31 | < 1e-12 | Pass |
| WSS power above h=6 | 1.77056e-31 | < 1e-12 | Pass |
| Thoracic force signed N=150 vs 180 relative L2 | 1.69495e-11 | < 1e-4 | Pass |
| Thoracic force exposure N=150 vs 180 relative L2 | 1.69218e-11 | < 1e-4 | Pass |
| Thoracic WSS N=150 vs 180 relative L2 | 2.53187e-13 | < 1e-6 | Pass |

## Six-artery observables

| Artery | Mean force exposure (pN) | Peak force exposure (pN) | RMS signed force (pN) | Mean |WSS| (Pa) | Peak |WSS| (Pa) |
|---|---:|---:|---:|---:|---:|
| Aortic Root | 10.3728 | 58.8887 | 17.6537 | 4.2344 | 12.3872 |
| Thoracic Aorta | 5.8465 | 30.6050 | 9.5416 | 3.2320 | 8.9277 |
| Femoral | 3.3903 | 10.8866 | 4.6886 | 2.5640 | 5.3108 |
| Carotid | 4.1181 | 18.2681 | 6.2497 | 2.7746 | 6.8889 |
| Iliac | 2.7823 | 8.8854 | 3.7962 | 2.3287 | 4.7994 |
| Brachial | 1.3765 | 4.6961 | 1.8964 | 1.6464 | 3.4842 |

The force exposure values are total magnitude-type control-volume observables. They are not anisotropy-specific effects. The anisotropy-specific increment remains separately exported.

## Exit decision

**Step 4 passes.** The standalone package reproduces all six arterial hydrodynamic cases and exports WSS and force observables with analytical, closure, spectral and grid-convergence guards. Step 5 may begin, but no membrane or Piezo1 coupling should be introduced until the original Piezo1 source model is independently reproduced.
