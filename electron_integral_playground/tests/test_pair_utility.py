
import pytest

import numpy as np

from electron_integral_playground.geometry_reader.xyz_reader import read_xyz_content
from electron_integral_playground.basis_reader.gaussian_basis_reader import read_basis
from electron_integral_playground.basis_utility import attach_basis_to_molecule
from electron_integral_playground.pair_utility import form_primitive_pair_list

# def test_pair_formation():
#     molecule = read_xyz_content(
#         """3

#         O 0 0 0
#         H 1 0 0
#         F -1.8 0.2 0
#         """.split("\n")
#     )
#     basis_set = read_basis("3-21g")
#     attach_basis_to_molecule(molecule, basis_set)
#     pair_list = form_primitive_pair_list(molecule, 1e-14)

#     print(pair_list)
#     raise NotImplementedError("Not finished")
