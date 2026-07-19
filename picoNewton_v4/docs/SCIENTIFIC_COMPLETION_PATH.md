# Scientific completion path

## Status of the historical notebook

`notebooks/picoNewton_v4_colab.ipynb` is retained unchanged as the archived first full run. It pins the historical package commit and must not be presented as the corrected publication workflow.

The corrected study is executed from the package with:

```bash
piconewton-v4-study \
  --package-root . \
  --profile full \
  --calibration configs/literature_calibration.json \
  --output /path/to/new_run
```

Every corrected run creates a new directory and never edits the notebook or an earlier run.

## Primary scientific rule

Piezo1 current is the primary endpoint. A hypothesis passes only when its RMS current difference exceeds the predeclared current threshold in at least the required number of arteries. Calcium is written to a separate exploratory table and cannot rescue a failed current decision.

## Required outputs

A corrected run writes:

- `model_outputs/`: the original numerical workflow outputs;
- `assessment/primary_current_decisions.csv`;
- `assessment/exploratory_calcium_screen.csv`;
- `assessment/primary_pathway_directionality.csv`;
- `assessment/signed_exposure_degeneracy_audit.csv`;
- `assessment/pressure_clipping_audit.csv`;
- `assessment/scientific_assessment.json`;
- `hydrodynamics/physical_forcing_waveforms.npz`;
- `hydrodynamics/hydrodynamic_diagnostics.csv`;
- `scientific_study_manifest.json` with SHA-256 hashes.

## Objective completion gates

A positive biological claim is prohibited unless all of the following are true:

1. structural numerical validation passes;
2. raw WSS, signed-force and exposure waveforms are archived;
3. a predeclared current comparison passes in the required number of arteries;
4. endpoint and transfer parameters are independently calibrated;
5. signed force and exposure are identifiable at the claimed aggregate endpoint, or the claim is explicitly spatial only;
6. primary pathways do not rely on an unresolved pressure ceiling;
7. the result survives held-out WSS-surrogate and matched-load controls;
8. independent review is complete.

The software records every gate in `scientific_assessment.json`. It keeps `claims_enabled` false by design. Enabling claims is a scientific review decision, not an automatic software state.

## Valid negative outcome

When the current endpoint fails the predeclared criterion, the run is still complete. Its correct status is `negative_under_current_parameterization`. Parameters must not be retuned after observing the result merely to manufacture a pass.
