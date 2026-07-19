import numpy as np
from piconewton_v4.piezo1 import Piezo1Parameters, generator_matrix, periodic_response

def test_detailed_balance_and_generator_conservation():
    p=Piezo1Parameters()
    assert p.r2 > 0
    q=generator_matrix(12.0,-40.0,p)
    assert np.max(np.abs(np.sum(q,axis=0))) < 1e-15

def test_periodic_probability_invariants():
    pressure=5.0+2.0*np.sin(2*np.pi*np.arange(64)/64)
    r=periodic_response(pressure,dt_s=1/1.2/64,gradmu_mv=-40.0)
    assert r['periodic_closure_error'] < 1e-9
    assert r['probability_sum_error'] < 1e-9
    assert r['minimum_probability'] >= -1e-10
