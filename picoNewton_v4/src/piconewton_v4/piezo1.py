"""Independent implementation of the Ogiermann et al. Piezo1 Markov model.

The state order is [Open, Closed, I1, I2]. Time is in milliseconds,
pressure is the non-negative source-model pressure magnitude in mmHg, and
gradmu is the electrochemical driving force in mV.

This module implements the published/committed CellML equations without any
hydrodynamic or membrane coupling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np
from scipy.linalg import expm


STATE_NAMES = ("P_Open", "P_Closed", "P_I1", "P_I2")


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


DEFAULT_INITIAL_STATE = np.array([0.0, 0.3, 0.1, 0.6], dtype=float)
CLOSED_INITIAL_STATE = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)


def transition_rates(
    pressure_mmhg: float,
    gradmu_mv: float,
    params: Piezo1Parameters = Piezo1Parameters(),
) -> dict[str, float]:
    p = float(pressure_mmhg)
    v = float(gradmu_mv)
    if not np.isfinite(p) or not np.isfinite(v):
        raise ValueError("pressure and gradmu must be finite")
    if p < 0:
        raise ValueError("source-model pressure magnitude must be non-negative")

    return {
        "O_to_I1": params.r1 * np.exp(params.ce1 * v / 140.0),
        "I1_to_O": params.r2 * np.exp(params.ce2 * v / 140.0 + params.cm2 * p / 70.0),
        "I1_to_C": params.r3 * np.exp(params.ce3 * v / 140.0),
        "C_to_I1": params.r4 * np.exp(params.ce4 * v / 140.0 + params.cm4 * p / 70.0),
        "C_to_O": params.r5 * np.exp(params.ce5 * v / 140.0),
        "O_to_C": params.r6 * np.exp(params.ce6 * v / 140.0),
        "I2_to_O": params.r7 * np.exp((params.ce7 * v / 140.0) * p / 70.0),
        "O_to_I2": params.r8 * np.exp(params.ce8 * v / 140.0),
    }


def generator_matrix(
    pressure_mmhg: float,
    gradmu_mv: float,
    params: Piezo1Parameters = Piezo1Parameters(),
) -> np.ndarray:
    r = transition_rates(pressure_mmhg, gradmu_mv, params)
    matrix = np.zeros((4, 4), dtype=float)

    def connect(source: int, destination: int, rate: float) -> None:
        matrix[destination, source] += rate
        matrix[source, source] -= rate

    connect(0, 2, r["O_to_I1"])
    connect(2, 0, r["I1_to_O"])
    connect(2, 1, r["I1_to_C"])
    connect(1, 2, r["C_to_I1"])
    connect(1, 0, r["C_to_O"])
    connect(0, 1, r["O_to_C"])
    connect(3, 0, r["I2_to_O"])
    connect(0, 3, r["O_to_I2"])
    return matrix


def validate_state(state: Sequence[float], tolerance: float = 1e-10) -> np.ndarray:
    values = np.asarray(state, dtype=float)
    if values.shape != (4,):
        raise ValueError("Piezo1 state must have shape (4,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("Piezo1 state must be finite")
    if np.min(values) < -tolerance:
        raise ValueError("Piezo1 state contains a negative probability")
    if abs(float(np.sum(values)) - 1.0) > tolerance:
        raise ValueError("Piezo1 probabilities must sum to one")
    return values


def propagate_constant(
    state: Sequence[float],
    duration_ms: float,
    pressure_mmhg: float,
    gradmu_mv: float,
    params: Piezo1Parameters = Piezo1Parameters(),
) -> np.ndarray:
    initial = validate_state(state)
    duration = float(duration_ms)
    if duration < 0 or not np.isfinite(duration):
        raise ValueError("duration_ms must be finite and non-negative")
    result = expm(generator_matrix(pressure_mmhg, gradmu_mv, params) * duration) @ initial
    result[np.abs(result) < 1e-15] = 0.0
    validate_state(result, tolerance=5e-10)
    return result


def sample_constant(
    state: Sequence[float],
    duration_ms: float,
    pressure_mmhg: float,
    gradmu_mv: float,
    *,
    dt_ms: float = 1.0,
    params: Piezo1Parameters = Piezo1Parameters(),
) -> tuple[np.ndarray, np.ndarray]:
    initial = validate_state(state)
    duration = float(duration_ms)
    dt = float(dt_ms)
    if duration < 0 or dt <= 0:
        raise ValueError("duration_ms must be non-negative and dt_ms positive")
    count = max(1, int(np.ceil(duration / dt)))
    time = np.linspace(0.0, duration, count + 1)
    matrix = generator_matrix(pressure_mmhg, gradmu_mv, params)
    states = np.stack([expm(matrix * value) @ initial for value in time], axis=0)
    return time, states


@dataclass(frozen=True)
class ProtocolSegment:
    duration_ms: float
    pressure_mmhg: float
    gradmu_mv: float
    label: str = ""


def simulate_protocol(
    segments: Iterable[ProtocolSegment],
    *,
    initial_state: Sequence[float] = CLOSED_INITIAL_STATE,
    dt_ms: float = 1.0,
    params: Piezo1Parameters = Piezo1Parameters(),
) -> dict[str, np.ndarray]:
    state = validate_state(initial_state)
    times: list[np.ndarray] = []
    states: list[np.ndarray] = []
    pressures: list[np.ndarray] = []
    voltages: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    offset = 0.0

    for index, segment in enumerate(segments):
        local_time, local_states = sample_constant(
            state,
            segment.duration_ms,
            segment.pressure_mmhg,
            segment.gradmu_mv,
            dt_ms=dt_ms,
            params=params,
        )
        if index:
            local_time = local_time[1:]
            local_states = local_states[1:]
        times.append(offset + local_time)
        states.append(local_states)
        pressures.append(np.full(local_time.shape, segment.pressure_mmhg, dtype=float))
        voltages.append(np.full(local_time.shape, segment.gradmu_mv, dtype=float))
        labels.append(np.full(local_time.shape, segment.label, dtype=object))
        state = local_states[-1]
        offset += segment.duration_ms

    if not times:
        raise ValueError("protocol requires at least one segment")
    state_matrix = np.concatenate(states, axis=0)
    probability_error = np.max(np.abs(np.sum(state_matrix, axis=1) - 1.0))
    if probability_error > 5e-10 or np.min(state_matrix) < -1e-10:
        raise RuntimeError("probability invariant violated")
    return {
        "time_ms": np.concatenate(times),
        "states": state_matrix,
        "P_Open": state_matrix[:, 0],
        "P_Closed": state_matrix[:, 1],
        "P_I1": state_matrix[:, 2],
        "P_I2": state_matrix[:, 3],
        "pressure_mmhg": np.concatenate(pressures),
        "gradmu_mv": np.concatenate(voltages),
        "label": np.concatenate(labels),
    }


def equilibrated_state(
    gradmu_mv: float,
    *,
    equilibration_ms: float = 20_000.0,
    initial_state: Sequence[float] = CLOSED_INITIAL_STATE,
    params: Piezo1Parameters = Piezo1Parameters(),
) -> np.ndarray:
    return propagate_constant(initial_state, equilibration_ms, 0.0, gradmu_mv, params)


def normalized_current_surrogate(
    open_probability: Sequence[float], gradmu_mv: Sequence[float] | float
) -> np.ndarray:
    """Ohmic source-model current surrogate up to one positive conductance factor."""
    return np.asarray(open_probability, dtype=float) * np.asarray(gradmu_mv, dtype=float)
