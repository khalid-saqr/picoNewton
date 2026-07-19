"""Piezo1 current and calcium-scale endpoints with explicit calibration status.

This module uses only the unchanged Step 5 public source-model API:
``Piezo1Parameters`` and vectorized generator matrices. It does not require or modify a
Step 5 periodic solver.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm
from .piezo1 import Piezo1Parameters, generator_matrices

FARADAY_C_PER_MOL = 96485.33212

@dataclass(frozen=True)
class EndpointParameters:
    channel_count: float = 1000.0
    single_channel_conductance_ps: float = 30.0
    membrane_voltage_mv: float = -40.0
    reversal_potential_mv: float = 0.0
    calcium_current_fraction: float = 0.05
    cell_volume_l: float = 2.0e-12
    calcium_clearance_time_s: float = 0.50
    current_detection_limit_pa: float = 1.0
    calcium_detection_limit_nm: float = 10.0
    calibration_status: str = "uncalibrated_engineering_reference"
    def validate(self) -> None:
        positive = [self.channel_count, self.single_channel_conductance_ps, self.cell_volume_l,
                    self.calcium_clearance_time_s, self.current_detection_limit_pa,
                    self.calcium_detection_limit_nm]
        if not np.all(np.isfinite(positive)) or np.min(positive) <= 0:
            raise ValueError("positive endpoint parameters required")
        if not 0.0 <= self.calcium_current_fraction <= 1.0:
            raise ValueError("calcium_current_fraction must lie in [0,1]")
    @property
    def calibrated(self) -> bool:
        return self.calibration_status == "experimentally_calibrated"

def _periodic_piezo1(pressure_mmhg: np.ndarray, *, dt_s: float, gradmu_mv: float,
                     p: Piezo1Parameters = Piezo1Parameters()) -> dict[str, np.ndarray | float]:
    pressure=np.asarray(pressure_mmhg,dtype=float)
    if pressure.ndim!=1 or pressure.size<8 or not np.all(np.isfinite(pressure)) or np.min(pressure)<0 or dt_s<=0:
        raise ValueError("invalid periodic Piezo1 input")
    matrices=expm(generator_matrices(pressure,float(gradmu_mv),p)*(1000.0*dt_s))
    monodromy=np.eye(4)
    for matrix in matrices: monodromy=matrix@monodromy
    system=monodromy-np.eye(4); rhs=np.zeros(4); system[-1,:]=1.0; rhs[-1]=1.0
    initial=np.linalg.solve(system,rhs); initial[np.abs(initial)<1e-14]=0.0
    if np.min(initial)<-1e-10: raise RuntimeError("periodic Piezo1 state is negative")
    initial=np.maximum(initial,0.0); initial/=np.sum(initial)
    states=np.empty((pressure.size,4),dtype=float); state=initial.copy()
    for index,matrix in enumerate(matrices): states[index]=state; state=matrix@state
    closure=float(np.max(np.abs(state-initial)))
    probability_error=float(np.max(np.abs(np.sum(states,axis=1)-1.0)))
    minimum=float(np.min(states))
    if closure>1e-9 or probability_error>1e-9 or minimum<-1e-10:
        raise RuntimeError("Piezo1 probability invariants failed")
    return {"states":states,"P_Open":states[:,0],"periodic_closure_error":closure,
            "probability_sum_error":probability_error,"minimum_probability":minimum,
            "monodromy_spectral_radius":float(np.max(np.abs(np.linalg.eigvals(monodromy))))}

def current_from_open_probability(p_open: np.ndarray, p: EndpointParameters) -> np.ndarray:
    p.validate(); probability = np.asarray(p_open, dtype=float)
    return (p.channel_count*p.single_channel_conductance_ps*probability*
            (p.membrane_voltage_mv-p.reversal_potential_mv)*1e-3)

def periodic_calcium_nm(current_pa: np.ndarray, *, dt_s: float, p: EndpointParameters) -> np.ndarray:
    p.validate(); current = np.asarray(current_pa, dtype=float)
    inward_a = np.maximum(-current, 0.0)*1e-12*p.calcium_current_fraction
    volume_m3 = p.cell_volume_l*1e-3
    influx_nm_s = inward_a/(2.0*FARADAY_C_PER_MOL*volume_m3)*1e6
    a = float(np.exp(-dt_s/p.calcium_clearance_time_s))
    b = influx_nm_s*p.calcium_clearance_time_s*(1.0-a)
    n = current.size
    weights = a**np.arange(n-1, -1, -1)
    c0 = float(np.sum(weights*b)/(1.0-a**n))
    calcium = np.empty(n, dtype=float); c = c0
    for i in range(n): calcium[i] = c; c = a*c+b[i]
    return calcium

def domain_endpoint(pressure_mmhg: np.ndarray, *, dt_s: float, gradmu_mv: float,
                    endpoint: EndpointParameters) -> dict[str, np.ndarray | float]:
    channel = _periodic_piezo1(pressure_mmhg, dt_s=dt_s, gradmu_mv=gradmu_mv)
    current = current_from_open_probability(np.asarray(channel["P_Open"]), endpoint)
    calcium = periodic_calcium_nm(current, dt_s=dt_s, p=endpoint)
    return {**channel, "current_pA": current, "calcium_nm": calcium,
            "charge_abs_fc_per_cycle": float(np.sum(np.abs(current))*dt_s*1e3),
            "calcium_auc_nm_s": float(np.sum(calcium)*dt_s)}

def aggregate_domains(apical: dict[str, np.ndarray | float], junctional: dict[str, np.ndarray | float],
                      *, apical_fraction: float) -> dict[str, np.ndarray | float]:
    if not 0 <= apical_fraction <= 1: raise ValueError("apical_fraction must lie in [0,1]")
    fa = apical_fraction; fj = 1.0-fa
    p_open = fa*np.asarray(apical["P_Open"])+fj*np.asarray(junctional["P_Open"])
    current = fa*np.asarray(apical["current_pA"])+fj*np.asarray(junctional["current_pA"])
    calcium = fa*np.asarray(apical["calcium_nm"])+fj*np.asarray(junctional["calcium_nm"])
    return {"P_Open": p_open, "current_pA": current, "calcium_nm": calcium,
            "periodic_closure_error": max(float(apical["periodic_closure_error"]), float(junctional["periodic_closure_error"])),
            "probability_sum_error": max(float(apical["probability_sum_error"]), float(junctional["probability_sum_error"])),
            "minimum_probability": min(float(apical["minimum_probability"]), float(junctional["minimum_probability"])),
            "charge_abs_fc_per_cycle": float(np.sum(np.abs(current))),
            "calcium_auc_nm_s": float(np.sum(calcium))}
