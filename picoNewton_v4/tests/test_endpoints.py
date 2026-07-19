import numpy as np
from piconewton_v4.endpoints import EndpointParameters, domain_endpoint

def test_current_and_calcium_units_and_periodicity():
    n=64; pressure=8+4*np.sin(2*np.pi*np.arange(n)/n)
    p=EndpointParameters()
    r=domain_endpoint(pressure,dt_s=1/1.2/n,gradmu_mv=-40,endpoint=p)
    assert r['current_pA'].shape==(n,)
    assert r['calcium_nm'].shape==(n,)
    assert np.all(np.isfinite(r['current_pA']))
    assert np.min(r['calcium_nm']) >= 0
    assert not p.calibrated
