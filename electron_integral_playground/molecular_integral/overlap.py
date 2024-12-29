
import numpy as np
import ctypes

from electron_integral_playground.data_structure import Molecule, PrimitivePairDataAngularList

libmolecular_integral = ctypes.cdll.LoadLibrary("libmolecular_integral.so")

def overlap(pair_data: PrimitivePairDataAngularList, molecule: Molecule) -> np.ndarray:
    n_ao = molecule.n_ao
    assert n_ao > 0
    S_matrix = np.zeros((n_ao, n_ao), dtype = np.float64, order = "C")

    for ij_angular, pair_data_of_angular in pair_data.items():
        n_pair = pair_data_of_angular.P_p.shape[0]
        assert n_pair > 0
        error_code = libmolecular_integral.overlap(
            ctypes.c_int(ij_angular[0]),
            ctypes.c_int(ij_angular[1]),
            pair_data_of_angular.A_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pair_data_of_angular.B_b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pair_data_of_angular.coefficient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pair_data_of_angular.i_ao_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            pair_data_of_angular.j_ao_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(n_pair),
            S_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n_ao),
            ctypes.c_bool(molecule.spherical_basis),
        )
        assert error_code == 0

    return S_matrix
