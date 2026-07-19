from pathlib import Path
import numpy as np
from piconewton_v4.types import HydrodynamicConfig, load_artery_cases
from piconewton_v4.hydrodynamics import compute_decomposition

def test_ground_truth_force_classes_present():
    root=Path(__file__).parents[1]
    case=load_artery_cases(root/'data'/'ground_truth_arteries.csv')[0]
    r=compute_decomposition(case,HydrodynamicConfig(32,64,16,0.1,0.1,1.0))
    assert np.min(r['force_exposure_anisotropic_n']) >= 0
    assert np.any(r['force_signed_anisotropic_n'] < 0)
    assert np.all(r['force_exposure_anisotropic_n'] >= np.abs(r['force_signed_anisotropic_n']) - 1e-18)
    assert r['wss_anisotropic_pa'].shape==(64,)
