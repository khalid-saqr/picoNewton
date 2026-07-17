# Step 5 — uncoupled Piezo1 source-model verification

## Scope

Step 5 implements the committed Ogiermann et al. four-state Piezo1 Markov model independently in Python and verifies it before any membrane, Lamb-force, or WSS coupling is introduced.

The source model is fixed to DOI `10.1113/JP288666`, Physiome workspace revision `8465ce0a385cdc40e2f79b001798d600e9f9e4d2`, and channel revision `7cbad662ee7cf9ed5e84b776bf3a6811f0ad1267`.

## Implemented source equations

The state vector is ordered as:

```text
[P_Open, P_Closed, P_I1, P_I2]
```

The implementation preserves all eight transition pathways, the exact CellML parameters, detailed-balance constraints for `r2`, `ce2`, and `cm2`, pressure in mmHg, voltage in mV, time in ms, and the publication's closed-state equilibration protocol.

Piecewise-constant protocols are propagated with a matrix exponential. Independent DOP853 integration provides a numerical cross-check.

## Accepted reproduction target

The primary source-model reproduction is the 500-ms, 70-mmHg voltage-clamp protocol over −80 to +50 mV. The verified model reproduces the reported qualitative voltage dependence: negative voltages cause substantially stronger inactivation than positive voltages.

## Repeated-pressure diagnostic

A repeated-pressure diagnostic is included at a fixed 1-s cadence with a declared 300-ms pressure-pulse width. It reproduces the expected ordering—negative-voltage trains desensitize more strongly than positive-voltage or alternating-voltage trains—but it is not claimed as an exact reproduction of Figure 4 because the channel-only CellML files do not encode the original pulse-width experiment definition.

## Validation gates

Step 5 passes only when:

1. committed CellML parameter values exactly match the Python implementation;
2. required source states and transition names are present;
3. every generator conserves probability;
4. state probabilities remain finite, nonnegative, and normalized;
5. matrix-exponential propagation agrees with DOP853;
6. the semigroup property passes;
7. positive voltage reduces 500-ms pressure-induced inactivation relative to negative voltage.

## Results

- Maximum generator column-sum error: `3.47e-18`.
- Maximum semigroup error: `1.08e-12`.
- Maximum matrix-exponential versus DOP853 error: `4.68e-11`.
- Maximum probability-sum error: `4.28e-12`.
- Minimum sampled probability: `4.89e-7`.
- Mean inactivation at negative voltages: `0.7276`.
- Mean inactivation at positive voltages: `0.1659`.

All primary Step 5 gates pass.

## Claim boundary

Step 5 verifies only the uncoupled source channel model. It does not show that the published near-wall force, WSS, membrane tension, or endothelial loading activates Piezo1. The physical interface remains a separate Step 6 model-development problem.
