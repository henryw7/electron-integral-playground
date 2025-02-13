
import numpy as np
import ctypes

from electron_integral_playground.data_structure import Molecule, PrimitivePairDataAngularList

libmolecular_integral = ctypes.cdll.LoadLibrary("libmolecular_integral.so")

def three_center(pair_data: PrimitivePairDataAngularList, auxiliary_pair_data: PrimitivePairDataAngularList, molecule: Molecule, omega: float = 0.0) -> np.ndarray:
    assert omega >= 0.0
    n_ao = molecule.n_ao
    assert n_ao > 0
    n_aux = molecule.n_aux
    assert n_aux > 0
    J3c_matrix = np.zeros((n_aux, n_ao, n_ao), dtype = np.float64, order = "C")

    for ij_angular, pair_data_ij in pair_data.items():
        i_angular, j_angular = ij_angular
        assert type(i_angular) is int
        assert type(j_angular) is int
        n_pair_ij = pair_data_ij.P_p.shape[0]
        assert n_pair_ij > 0
        for k_angular, pair_data_k in auxiliary_pair_data.items():
            assert type(k_angular) is int
            n_aux_primitive_k = pair_data_k.A_a.shape[0]
            assert n_aux_primitive_k > 0

            error_code = libmolecular_integral.three_center(
                ctypes.c_int(i_angular),
                ctypes.c_int(j_angular),
                ctypes.c_int(k_angular),
                pair_data_ij.P_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                pair_data_ij.A_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                pair_data_ij.B_b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                pair_data_ij.coefficient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                pair_data_ij.i_ao_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                pair_data_ij.j_ao_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                ctypes.c_int(n_pair_ij),
                pair_data_k.A_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                pair_data_k.coefficient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                pair_data_k.i_ao_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                ctypes.c_int(n_aux_primitive_k),
                J3c_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(n_ao),
                ctypes.c_int(n_aux),
                ctypes.c_bool(molecule.spherical_basis),
                ctypes.c_double(omega),
            )
            assert error_code == 0

    return J3c_matrix
