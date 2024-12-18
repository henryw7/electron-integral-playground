
import pytest

import numpy as np

from electron_integral_playground.geometry_reader.xyz_reader import read_xyz_content
from electron_integral_playground.basis_reader.gaussian_basis_reader import read_basis
from electron_integral_playground.basis_utility import normalize_shell, attach_basis_to_molecule, assign_ao_index_to_shell, AtomicOrbitalOrder

# def test_basis_normalization_multiple_primitives():
#     shell = GaussianShell(
#         angular = 0,
#         spherical = True,
#         i_ao_start = -1,
#         i_atom = -1,
#         primitive_exponents = np.array([0.3683820000E+02, 0.5481720000E+01, 0.1113270000E+01]),
#         primitive_coefficients = np.array([0.6966866381E-01, 0.3813463493E+00, 0.6817026244E+00]),
#     )
#     normalize_shell(shell)
#     print("S shell")
#     print(shell)

#     normalize_shell(shell)
#     print("S shell again")
#     print(shell)

#     shell.angular = 1
#     normalize_shell(shell)
#     print("P shell")
#     print(shell)

#     shell.angular = 2
#     normalize_shell(shell)
#     print("D shell")
#     print(shell)

#     shell.angular = 3
#     normalize_shell(shell)
#     print("F shell")
#     print(shell)

#     shell.angular = 3
#     normalize_shell(shell)
#     print("F shell again")
#     print(shell)

#     raise NotImplementedError("Reference answer not available yet")

# def test_basis_normalization_one_primitive():
#     shell = GaussianShell(
#         angular = 0,
#         spherical = True,
#         i_ao_start = -1,
#         i_atom = -1,
#         primitive_exponents = np.array([0.1243280000E+00]),
#         primitive_coefficients = np.array([1.0]),
#     )
#     normalize_shell(shell)
#     print("S shell")
#     print(shell)

#     normalize_shell(shell)
#     print("S shell again")
#     print(shell)

#     shell.angular = 1
#     normalize_shell(shell)
#     print("P shell")
#     print(shell)

#     shell.angular = 2
#     normalize_shell(shell)
#     print("D shell")
#     print(shell)

#     shell.angular = 3
#     normalize_shell(shell)
#     print("F shell")
#     print(shell)

#     shell.angular = 3
#     normalize_shell(shell)
#     print("F shell again")
#     print(shell)

#     raise NotImplementedError("Reference answer not available yet")

def test_basis_assignment_low_angular():
    molecule = read_xyz_content(
        """3

        O 0 0 0
        H 1 0 0
        F -0.8 0.2 0
        """.split("\n")
    )
    basis_set = read_basis("3-21g")
    n_ao = attach_basis_to_molecule(molecule, basis_set)

    reference_angular_list = [ 0,0,1,0,1, 0,0, 0,0,1,0,1 ]
    reference_i_atom_list = [ 0,0,0,0,0, 1,1, 2,2,2,2,2 ]
    reference_i_ao_start_list = [ 0,1,2,5,6, 9,10, 11,12,13,16,17 ]
    reference_n_ao = 20

    for i_shell, shell in enumerate(molecule.basis_shells):
        assert shell.angular == reference_angular_list[i_shell]
        assert shell.i_atom == reference_i_atom_list[i_shell]
        assert shell.i_ao_start == reference_i_ao_start_list[i_shell]
    assert n_ao == reference_n_ao

    n_ao = assign_ao_index_to_shell(molecule, AtomicOrbitalOrder.ANGULAR_LEADING)

    reference_angular_list = [ 0,0,0, 0,0, 0,0,0, 1,1, 1,1 ]
    reference_i_atom_list = [ 0,0,0, 1,1, 2,2,2, 0,0, 2,2 ]
    reference_i_ao_start_list = [ 0,1,2, 3,4, 5,6,7, 8,11, 14,17 ]
    reference_n_ao = 20

    for i_shell, shell in enumerate(molecule.basis_shells):
        assert shell.angular == reference_angular_list[i_shell]
        assert shell.i_atom == reference_i_atom_list[i_shell]
        assert shell.i_ao_start == reference_i_ao_start_list[i_shell]
    assert n_ao == reference_n_ao

def test_basis_assignment_high_angular():
    molecule = read_xyz_content(
        """3

        O 0 0 0
        H 1 0 0
        H -0.8 0.2 0
        """.split("\n")
    )
    basis_set = read_basis("def2-tzvp")
    n_ao = attach_basis_to_molecule(molecule, basis_set)

    reference_angular_list = [ 0,0,0,0,0,1,1,1,2,2,3, 0,0,0,1, 0,0,0,1 ]
    reference_i_atom_list = [ 0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1, 2,2,2,2 ]
    reference_i_ao_start_list = [ 0,1,2,3,4,5,8,11,14,19,24, 31,32,33,34, 37,38,39,40 ]
    reference_n_ao = 43

    for i_shell, shell in enumerate(molecule.basis_shells):
        assert shell.angular == reference_angular_list[i_shell]
        assert shell.i_atom == reference_i_atom_list[i_shell]
        assert shell.i_ao_start == reference_i_ao_start_list[i_shell]
    assert n_ao == reference_n_ao

    for shell in molecule.basis_shells:
        shell.spherical = False
    n_ao = assign_ao_index_to_shell(molecule, AtomicOrbitalOrder.ANGULAR_LEADING)

    reference_angular_list = [ 0,0,0,0,0, 0,0,0, 0,0,0, 1,1,1, 1, 1, 2,2, 3 ]
    reference_i_atom_list = [ 0,0,0,0,0, 1,1,1, 2,2,2, 0,0,0, 1, 2, 0,0, 0 ]
    reference_i_ao_start_list = [ 0,1,2,3,4, 5,6,7, 8,9,10, 11,14,17, 20, 23, 26,32, 38 ]
    reference_n_ao = 48

    for i_shell, shell in enumerate(molecule.basis_shells):
        assert shell.angular == reference_angular_list[i_shell]
        assert shell.i_atom == reference_i_atom_list[i_shell]
        assert shell.i_ao_start == reference_i_ao_start_list[i_shell]
    assert n_ao == reference_n_ao
