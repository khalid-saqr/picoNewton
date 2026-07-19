import numpy as np
from piconewton_v4.piezo1 import generator_matrix, generator_matrices


def test_vectorized_generators_match_scalar_source_equations():
    pressure = np.array([0.0, 7.5, 20.0])
    batch = generator_matrices(pressure, -40.0)
    scalar = np.stack([generator_matrix(float(value), -40.0) for value in pressure])
    assert np.max(np.abs(batch - scalar)) < 1e-15
    assert np.max(np.abs(np.sum(batch, axis=-2))) < 1e-15
