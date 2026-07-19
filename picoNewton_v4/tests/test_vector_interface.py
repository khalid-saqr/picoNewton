import numpy as np
from piconewton_v4.vector_interface import VectorInterfaceParameters, vector_membrane_state, validate_passivity, elastic_limit

def test_vector_interface_keeps_force_classes_and_directions_separate():
    n=128; t=np.arange(n)/n
    wss=2*np.sin(2*np.pi*t)
    signed=10e-12*np.sin(2*np.pi*t+0.3)
    exposure=np.abs(signed)
    p=VectorInterfaceParameters()
    r=vector_membrane_state(wall_shear_pa=wss,signed_force_n=signed,force_exposure_n=exposure,dt_s=1/1.2/n,p=p)
    assert not r['exposure_used_as_signed_load']
    assert np.min(r['normal_exposure_traction_pa']) >= 0
    assert np.any(r['normal_signed_traction_pa'] < 0) and np.any(r['normal_signed_traction_pa'] > 0)
    assert np.all(r['apical_pressure_mmhg'] >= 0)
    assert np.all(r['junctional_pressure_mmhg'] >= 0)

def test_fast_slow_passivity_and_elastic_limit():
    p=VectorInterfaceParameters()
    assert validate_passivity(p.tangential)['passed']
    assert validate_passivity(p.normal)['passed']
    e=elastic_limit(p)
    assert e.tangential.is_elastic and e.normal.is_elastic
