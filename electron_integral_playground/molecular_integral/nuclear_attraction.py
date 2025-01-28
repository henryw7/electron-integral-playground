
import numpy as np
import ctypes

from electron_integral_playground.data_structure import Molecule, PrimitivePairDataAngularList

libmolecular_integral = ctypes.cdll.LoadLibrary("libmolecular_integral.so")

def nuclear_attraction(pair_data: PrimitivePairDataAngularList, molecule: Molecule, charge_position: np.ndarray) -> np.ndarray:
    n_ao = molecule.n_ao
    assert n_ao > 0
    assert charge_position.ndim == 2
    assert charge_position.shape[1] == 3
    n_charge = charge_position.shape[0]
    assert n_charge > 0
    if charge_position.dtype is not np.float64:
        charge_position = charge_position.astype(np.float64)
    if not charge_position.flags["C_CONTIGUOUS"]:
        charge_position = charge_position.copy(order = "C")
    V_tensor = np.zeros((n_ao, n_ao, n_charge), dtype = np.float64, order = "C")

    for ij_angular, pair_data_of_angular in pair_data.items():
        n_pair = pair_data_of_angular.P_p.shape[0]
        assert n_pair > 0
        error_code = libmolecular_integral.nuclear_attraction(
            ctypes.c_int(ij_angular[0]),
            ctypes.c_int(ij_angular[1]),
            pair_data_of_angular.P_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pair_data_of_angular.A_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pair_data_of_angular.B_b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pair_data_of_angular.coefficient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pair_data_of_angular.i_ao_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            pair_data_of_angular.j_ao_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(n_pair),
            charge_position.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n_charge),
            V_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n_ao),
            ctypes.c_bool(molecule.spherical_basis),
        )
        assert error_code == 0

    return V_tensor
