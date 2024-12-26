
import pytest

import numpy as np

from electron_integral_playground.units import LengthUnits
from electron_integral_playground.geometry_reader.xyz_reader import read_xyz_content
from electron_integral_playground.basis_reader.gaussian_basis_reader import read_basis, read_basis_content
from electron_integral_playground.basis_utility import attach_basis_to_molecule
from electron_integral_playground.pair_utility import form_primitive_pair_list, form_primitive_pair_data
from electron_integral_playground.molecular_integral.overlap import overlap

def test_overlap_low_angular():
    molecule = read_xyz_content(
        """3

        O 0 0 0
        H 1 0 0
        F -0.8 0.2 0
        """.split("\n")
    )
    # molecule = read_xyz_content(
    #     """2

    #     H 0 0 0
    #     F 1 0.1 0
    #     """.split("\n")
    # )
    basis_set = read_basis("sto-3g")
#     basis_set = read_basis_content("""
# H     0
# S    2   1.00
#       0.1831915800D+00       1.0000000
#       1.1831915800E+00       1.0000000
# ****
                                   
# """.split("\n"))
    attach_basis_to_molecule(molecule, basis_set)
    pair_list = form_primitive_pair_list(molecule, 1e-14)
    pair_data_list = form_primitive_pair_data(molecule, pair_list)
    test_S = overlap(pair_data_list, molecule)

    np.set_printoptions(linewidth = 1000000, threshold = 1000000, precision = 7, suppress = True)
    print(test_S)

    raise NotImplementedError("Not finished")
