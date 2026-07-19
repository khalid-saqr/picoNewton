"""Exact four-state Ogiermann Piezo1 Markov source model.

State order: [P_Open, P_Closed, P_I1, P_I2]. Pressure is in mmHg,
voltage/electrochemical gradient in mV, and generator rates in ms^-1.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm

@dataclass(frozen=True)
class Piezo1Parameters:
    r1: float = 0.02634342
    r3: float = 0.0008000712
    r4: float = 0.008945307
    r5: float = 1.3529515e-5
    r6: float = 6.206403e-5
    r7: float = 6.603723e-6
    r8: float = 0.00079357607
    ce1: float = -2.3753126
    ce3: float = -5.6010337
    ce4: float = 0.70629007
    cm4: float = -12.101605
    ce5: float = 6.46389
    ce6: float = -1.2173947
    ce7: float = 8.739162
    ce8: float = -3.364573
    @property
    def r2(self) -> float:
        return self.r1 * self.r3 * self.r5 / (self.r4 * self.r6)
    @property
    def ce2(self) -> float:
        return self.ce1 + self.ce3 + self.ce5 - self.ce4 - self.ce6
    @property
    def cm2(self) -> float:
        return -self.cm4

def transition_rates(pressure_mmhg: float | np.ndarray, gradmu_mv: float | np.ndarray,
                     p: Piezo1Parameters = Piezo1Parameters()) -> dict[str, np.ndarray]:
    pressure = np.asarray(pressure_mmhg, dtype=float)
    voltage = np.asarray(gradmu_mv, dtype=float)
    pressure, voltage = np.broadcast_arrays(pressure, voltage)
    return {
        "O_to_I1": p.r1 * np.exp(p.ce1 * voltage / 140.0),
        "I1_to_O": p.r2 * np.exp(p.ce2 * voltage / 140.0 + p.cm2 * pressure / 70.0),
        "I1_to_C": p.r3 * np.exp(p.ce3 * voltage / 140.0),
        "C_to_I1": p.r4 * np.exp(p.ce4 * voltage / 140.0 + p.cm4 * pressure / 70.0),
        "C_to_O": p.r5 * np.exp(p.ce5 * voltage / 140.0),
        "O_to_C": p.r6 * np.exp(p.ce6 * voltage / 140.0),
        "I2_to_O": p.r7 * np.exp((p.ce7 * voltage / 140.0) * pressure / 70.0),
        "O_to_I2": p.r8 * np.exp(p.ce8 * voltage / 140.0),
    }

def generator_matrix(pressure_mmhg: float, gradmu_mv: float,
                     p: Piezo1Parameters = Piezo1Parameters()) -> np.ndarray:
    rates = {k: float(v) for k, v in transition_rates(pressure_mmhg, gradmu_mv, p).items()}
    q = np.zeros((4, 4), dtype=float)
    def connect(source: int, destination: int, rate: float) -> None:
        q[destination, source] += rate; q[source, source] -= rate
    connect(0, 2, rates["O_to_I1"]); connect(2, 0, rates["I1_to_O"])
    connect(2, 1, rates["I1_to_C"]); connect(1, 2, rates["C_to_I1"])
    connect(1, 0, rates["C_to_O"]); connect(0, 1, rates["O_to_C"])
    connect(3, 0, rates["I2_to_O"]); connect(0, 3, rates["O_to_I2"])
    return q



def generator_matrices(pressure_mmhg: np.ndarray, gradmu_mv: float | np.ndarray,
                       p: Piezo1Parameters = Piezo1Parameters()) -> np.ndarray:
    """Vectorized generator matrices with shape ``(..., 4, 4)``."""
    rates = transition_rates(pressure_mmhg, gradmu_mv, p)
    shape = np.broadcast_shapes(np.shape(pressure_mmhg), np.shape(gradmu_mv))
    q = np.zeros((*shape, 4, 4), dtype=float)

    def connect(source: int, destination: int, rate: np.ndarray) -> None:
        q[..., destination, source] += rate
        q[..., source, source] -= rate

    connect(0, 2, rates["O_to_I1"])
    connect(2, 0, rates["I1_to_O"])
    connect(2, 1, rates["I1_to_C"])
    connect(1, 2, rates["C_to_I1"])
    connect(1, 0, rates["C_to_O"])
    connect(0, 1, rates["O_to_C"])
    connect(3, 0, rates["I2_to_O"])
    connect(0, 3, rates["O_to_I2"])
    return q

def _periodic_fixed_state(matrices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    monodromy = np.eye(4)
    for m in matrices:
        monodromy = m @ monodromy
    system = monodromy - np.eye(4); rhs = np.zeros(4)
    system[-1, :] = 1.0; rhs[-1] = 1.0
    state = np.linalg.solve(system, rhs)
    state[np.abs(state) < 1e-14] = 0.0
    if np.min(state) < -1e-10:
        raise RuntimeError("periodic Piezo1 state is negative")
    state = np.maximum(state, 0.0); state /= np.sum(state)
    return state, monodromy

def periodic_response(pressure_mmhg: np.ndarray, *, dt_s: float, gradmu_mv: float,
                      p: Piezo1Parameters = Piezo1Parameters()) -> dict[str, np.ndarray | float]:
    pressure = np.asarray(pressure_mmhg, dtype=float)
    if pressure.ndim != 1 or pressure.size < 8 or not np.all(np.isfinite(pressure)) or np.min(pressure) < 0:
        raise ValueError("pressure must be finite, nonnegative and one-dimensional")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    matrices = expm(generator_matrices(pressure, gradmu_mv, p) * (1000.0 * dt_s))
    initial, monodromy = _periodic_fixed_state(matrices)
    states = np.empty((pressure.size, 4), dtype=float); state = initial.copy()
    for i, matrix in enumerate(matrices):
        states[i] = state; state = matrix @ state
    closure = float(np.max(np.abs(state-initial)))
    probability_error = float(np.max(np.abs(np.sum(states, axis=1)-1.0)))
    minimum = float(np.min(states))
    if closure > 1e-9 or probability_error > 1e-9 or minimum < -1e-10:
        raise RuntimeError("Piezo1 probability invariants failed")
    return {
        "states": states, "P_Open": states[:, 0],
        "periodic_closure_error": closure, "probability_sum_error": probability_error,
        "minimum_probability": minimum,
        "monodromy_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(monodromy)))),
    }
