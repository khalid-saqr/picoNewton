# Step 3 stress-test

## Decision

**PASS — proceed to Step 4.**

Step 3 is scientifically and computationally closed after three defects found during stress-testing were corrected: isotropic exposure is exported explicitly, convergence is evaluated on the anisotropic excess as well as the total force, and Step 3 is no longer imported eagerly by the package initializer.

## Over-simplification

**PASS.** Step 3 does more than rerun six cases. It independently retains anisotropic, isotropic, signed, exposure, signed-excess, exposure-difference, verified, and historical outputs; validates the isotropic analytical solution; checks mechanics closure; and tests the small excess against numerical resolution.

It does not attempt the Step 4 perturbation hierarchy prematurely.

## Overcomplication

**PASS.** No solver is copied into the successor package. The Step 3 layer calls the frozen parent hydrodynamic interface and adds only orchestration, metrics, convergence, exports, gates, and lineage control. Step 3 is accessed through its dedicated `continuity` module and CLI, so Step 2 bootstrap imports remain isolated from the later scientific layer.

No mechanosensor, membrane, ion-channel, compliant-wall, geometry, Sobol, machine-learning, kernel, susceptibility, or threshold-inversion calculation is included.

## Feasibility

**PASS.** The complete publication profile covers six arteries, three solver states per artery, and declared convergence variants using one-dimensional harmonic systems. The observed runtime and data volume are small relative to later experiment stages. All local executable tests passed.

## Parent-model fidelity

**PASS.** The six artery definitions, native pressure-gradient scales, six harmonics, fluid properties, endothelial control volume, anisotropy state, verified solver mode, and historical mode remain inherited from the frozen parent source.

Computed Womersley numbers agree with the published inventory to within the predeclared 0.5% tolerance.

## Scientific value

**PASS.** Step 3 establishes a result that is essential for the successor:

- the total signed near-wall Lamb-force is dominated by the isotropic reference;
- the anisotropy-induced signed RMS increment is approximately 0.99–1.08% of the isotropic RMS across the six arteries;
- the increment remains far above numerical convergence error;
- the verified and historical nonlinear evaluations differ substantially and must not be mixed.

This justifies Step 4's perturbative focus on the constitutive excess rather than on the total Lamb-force proxy.

## Six arteries

**PASS.** All six native cases are included and no artery is dropped from the summaries, waveform archive, historical comparison, convergence table, or gates.

The arteries are not yet crossed with one another's waveforms; that remains a later experiment after the perturbative and kernel machinery exists.

## Comprehensive result coverage

Step 3 exports:

- verified anisotropic total signed force and exposure;
- verified isotropic signed force and exposure;
- signed anisotropic excess and exposure difference;
- directional and high-harmonic metrics;
- historical-mode waveforms and discrepancies;
- isotropic analytical validation;
- mechanics closure;
- radial, temporal, and quadrature convergence;
- checksummed waveform arrays and tables;
- a fail-closed Step 3 gate.

## Defects corrected during stress-testing

1. The first draft omitted an explicit isotropic exposure export. It is now retained in the summary and waveform archive.
2. The first convergence gate tested the much larger total force but not the small constitutive excess. Both are now independently required to converge within 1%.
3. The first package initializer imported Step 3 eagerly, which could load the broader parent package during a Step 2 bootstrap import. Step 3 is now loaded only through its dedicated module or CLI.

## Residual risks

1. The roughly 1% constitutive increment is numerically secure but scientifically modest. Step 4 must demonstrate the predicted perturbative structure rather than merely restate this percentage.
2. Historical-mode discrepancies are large. The manuscript must describe that path as computational lineage only.
3. Similar signed and exposure metrics in the native near-wall layer must not be generalized beyond the tested control-volume geometry.
4. Step 3 does not yet determine whether the excess scales as \(\varepsilon^2\). That is the explicit purpose of Step 4.

## Recommendation

Start Step 4. No Step 3 revision is required before implementing the weak-anisotropy perturbation hierarchy.
