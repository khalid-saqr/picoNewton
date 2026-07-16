# Preliminary reduced-profile dry run

The Step-6 dry run used `N=70`, 256 time points and 64 near-wall quadrature
nodes. It verified end-to-end implementation but is not a publication result.

Preliminary gates:

- E1 core cross-artery detectability: failed.
- E2 held-out WSS nonredundancy: failed.
- E3 contiguous parameter support: passed.
- E4 directional specificity: passed.
- E5 high-harmonic specificity: failed.
- E6 anisotropy-specific detectability: failed.
- E7 full-range robustness: failed.
- E8 model-class transparency: passed.

These outcomes are fixtures for regression testing only. The publication
profile recomputes every gate and must preserve negative findings unless the
locked high-resolution analysis changes them.
