# Output reference

## `six_artery_summary.csv`

One row per artery and pathway. Fields include:

- mean and dynamic range of Piezo1 open probability;
- RMS, peak, dynamic range, and cycle charge of Piezo1 current;
- mean, peak, dynamic range, and cycle integral of calcium-scale response;
- apical and junctional current metrics;
- spatial polarity index;
- harmonic power fraction;
- peak equivalent pressures;
- force-angle range;
- probability conservation diagnostics.

## `hypothesis_effects.csv`

Pairwise endpoint differences for activation, H3a, H3b, H4, H5, H7, and matched-load controls.

Primary fields:

- `current_rms_difference_pa`
- `current_peak_difference_pa`
- `current_relative_rms`
- `calcium_rms_difference_nm`
- `calcium_peak_difference_nm`
- `calcium_relative_rms`

## `loao_vector_surrogates.csv`

Leave-one-artery-out causal WSS-surrogate parameters and held-out raw-traction agreement.

## `artery_feature_distances.csv`

Pairwise standardized distances between artery response-feature vectors for signed, exposure, and combined vector pathways.

## `waveforms.npz`

Compressed arrays for each artery, pathway, endpoint, and membrane-domain pressure. Load with:

```python
import numpy as np
arrays = np.load("waveforms.npz")
print(arrays.files)
```

## `validation.json`

Structural validation, passivity, probability conservation, force-class separation, calibration audit, and numerical profile.

## `manifest.json`

Run configuration, model parameters, calibration audit, output checksums, and completion timestamp.

## `hypothesis_decisions.csv`

Created by the notebook when explicit current and calcium thresholds are applied. The thresholds are copied to the run provenance.

## Figures

The notebook generates:

- six-artery RMS current comparison;
- H3 current-difference chart;
- one current-waveform figure per artery.
