
import numpy as np
import ctypes

from electron_integral_playground.data_structure import Molecule, PrimitivePairDataAngularList

libmolecular_integral = ctypes.cdll.LoadLibrary("libmolecular_integral.so")

def two_center(pair_data: PrimitivePairDataAngularList, molecule: Molecule, omega: float = 0.0) -> np.ndarray:
    assert omega >= 0.0
    n_aux = molecule.n_aux
    assert n_aux > 0
    J2c_matrix = np.zeros((n_aux, n_aux), dtype = np.float64, order = "C")

    for i_angular, pair_data_i in pair_data.items():
        n_aux_primitive_i = pair_data_i.A_a.shape[0]
        assert n_aux_primitive_i > 0
        for j_angular, pair_data_j in pair_data.items():
            n_aux_primitive_j = pair_data_j.A_a.shape[0]
            assert n_aux_primitive_j > 0

            if (i_angular > j_angular):
                continue

            error_code = libmolecular_integral.two_center(
                ctypes.c_int(i_angular),
                ctypes.c_int(j_angular),
                pair_data_i.A_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                pair_data_i.coefficient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                pair_data_i.i_ao_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                ctypes.c_int(n_aux_primitive_i),
                pair_data_j.A_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                pair_data_j.coefficient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                pair_data_j.i_ao_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                ctypes.c_int(n_aux_primitive_j),
                J2c_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(n_aux),
                ctypes.c_bool(molecule.spherical_basis),
                ctypes.c_double(omega),
            )
            assert error_code == 0

    return J2c_matrix
