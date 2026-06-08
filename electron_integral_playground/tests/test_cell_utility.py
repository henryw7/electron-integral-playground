
import pytest

import numpy as np

from electron_integral_playground.cell_utility import get_cell_from_lattice_constant

lattice_vector_threshold = 1.01e-5

def test_read_lattice_constant():
    a = 10.0
    b = 11.0
    c = 12.0
    alpha = 90 * np.pi / 180.0
    beta  = 90 * np.pi / 180.0
    gamma = 90 * np.pi / 180.0
    ref_R = np.array([[10.0, 0, 0], [0, 11.0, 0], [0, 0, 12.0]])

    cell = get_cell_from_lattice_constant(a, b, c, alpha, beta, gamma)
    assert cell.dimension == 3
    test_R = cell.lattice_vectors
    np.testing.assert_allclose(test_R, ref_R, rtol = 0, atol = 1e-14)

    # From here on, reference obtained from Avogadro 1.99.0

    a = 5.5220
    b = 5.4396
    c = 7.6726
    alpha =  90.00 * np.pi / 180.0
    beta  = 110.55 * np.pi / 180.0
    gamma =  90.00 * np.pi / 180.0
    ref_R = np.array([
        [ 5.52200,  0.00000,  0.00000],
        [-0.00000,  5.43960,  0.00000],
        [-2.69327, -0.00000,  7.18436],
    ])

    cell = get_cell_from_lattice_constant(a, b, c, alpha, beta, gamma)
    assert cell.dimension == 3
    test_R = cell.lattice_vectors
    np.testing.assert_allclose(test_R, ref_R, rtol = 0, atol = lattice_vector_threshold)

    a = 7.0
    b = 8.0
    c = 9.0
    alpha = 60 * np.pi / 180.0
    beta  = 60 * np.pi / 180.0
    gamma = 60 * np.pi / 180.0
    ref_R = np.array([
        [ 7.00000,  0.00000,  0.00000],
        [ 4.00000,  6.92820,  0.00000],
        [ 4.50000,  2.59808,  7.34847],
    ])

    cell = get_cell_from_lattice_constant(a, b, c, alpha, beta, gamma)
    assert cell.dimension == 3
    test_R = cell.lattice_vectors
    np.testing.assert_allclose(test_R, ref_R, rtol = 0, atol = lattice_vector_threshold)

    a = 7.0
    b = 8.0
    c = 9.0
    alpha =  60 * np.pi / 180.0
    beta  = 140 * np.pi / 180.0
    gamma = 100 * np.pi / 180.0
    ref_R = np.array([
        [  7.00000,  0.00000,  0.00000],
        [ -1.38919,  7.87846,  0.00000],
        [ -6.89440,  3.35375,  4.71377],
    ])

    cell = get_cell_from_lattice_constant(a, b, c, alpha, beta, gamma)
    assert cell.dimension == 3
    test_R = cell.lattice_vectors
    np.testing.assert_allclose(test_R, ref_R, rtol = 0, atol = lattice_vector_threshold)
