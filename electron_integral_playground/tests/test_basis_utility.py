
import pytest

import numpy as np
from copy import deepcopy

from electron_integral_playground.data_structure import GaussianShell
from electron_integral_playground.geometry_reader.xyz_reader import read_xyz_content
from electron_integral_playground.basis_reader.gaussian_basis_reader import read_basis
from electron_integral_playground.basis_utility import normalize_shell, attach_basis_to_molecule, assign_ao_index_to_shell, AtomicOrbitalOrder

normalization_threshold = 1e-14

def test_basis_normalization_multiple_primitives():
    shell = GaussianShell(
        angular = 0,
        i_ao_start = -1,
        i_atom = -1,
        primitive_exponents = np.array([0.3683820000E+02, 0.5481720000E+01, 0.1113270000E+01]),
        primitive_coefficients = np.array([0.6966866381E-01, 0.3813463493E+00, 0.6817026244E+00]),
    )

    s_shell = deepcopy(shell)
    normalize_shell(s_shell)
    np.testing.assert_allclose(s_shell.primitive_exponents, np.array([0.3683820000E+02, 0.5481720000E+01, 0.1113270000E+01]), atol = 0)
    np.testing.assert_allclose(s_shell.primitive_coefficients, np.array([0.7424571167056886, 0.9736824041834441, 0.5265691816179123]), atol = normalization_threshold)

    p_shell = deepcopy(shell)
    p_shell.angular = 1
    normalize_shell(p_shell)
    np.testing.assert_allclose(p_shell.primitive_exponents, np.array([0.3683820000E+02, 0.5481720000E+01, 0.1113270000E+01]), atol = 0)
    np.testing.assert_allclose(p_shell.primitive_coefficients, np.array([9.53466870700217, 4.823483291037769, 1.175548964021699]), atol = normalization_threshold)

    d_shell = deepcopy(shell)
    d_shell.angular = 2
    normalize_shell(d_shell)
    np.testing.assert_allclose(d_shell.primitive_exponents, np.array([0.3683820000E+02, 0.5481720000E+01, 0.1113270000E+01]), atol = 0)
    np.testing.assert_allclose(d_shell.primitive_coefficients, np.array([69.77322364641736, 13.616114130201483, 1.495460534593661]), atol = normalization_threshold)

    f_shell = deepcopy(shell)
    f_shell.angular = 3
    normalize_shell(f_shell)
    np.testing.assert_allclose(f_shell.primitive_exponents, np.array([0.3683820000E+02, 0.5481720000E+01, 0.1113270000E+01]), atol = 0)
    np.testing.assert_allclose(f_shell.primitive_coefficients, np.array([391.7035485493139, 29.48703626431445, 1.4594685003804202]), atol = normalization_threshold)

def test_basis_normalization_one_primitive():
    shell = GaussianShell(
        angular = 0,
        i_ao_start = -1,
        i_atom = -1,
        primitive_exponents = np.array([0.1243280000E+00]),
        primitive_coefficients = np.array([1.0]),
    )

    s_shell = deepcopy(shell)
    normalize_shell(s_shell)
    np.testing.assert_allclose(s_shell.primitive_exponents, np.array([0.1243280000E+00]), atol = 0)
    np.testing.assert_allclose(s_shell.primitive_coefficients, np.array([0.1492233559500336]), atol = normalization_threshold)

    p_shell = deepcopy(shell)
    p_shell.angular = 1
    normalize_shell(p_shell)
    np.testing.assert_allclose(p_shell.primitive_exponents, np.array([0.1243280000E+00]), atol = 0)
    np.testing.assert_allclose(p_shell.primitive_coefficients, np.array([0.1052328353933318]), atol = normalization_threshold)

    d_shell = deepcopy(shell)
    d_shell.angular = 2
    normalize_shell(d_shell)
    np.testing.assert_allclose(d_shell.primitive_exponents, np.array([0.1243280000E+00]), atol = 0)
    np.testing.assert_allclose(d_shell.primitive_coefficients, np.array([0.0428454900225391]), atol = normalization_threshold)

    f_shell = deepcopy(shell)
    f_shell.angular = 3
    normalize_shell(f_shell)
    np.testing.assert_allclose(f_shell.primitive_exponents, np.array([0.1243280000E+00]), atol = 0)
    np.testing.assert_allclose(f_shell.primitive_coefficients, np.array([0.0135124649803557]), atol = normalization_threshold)

def test_basis_assignment_low_angular():
    molecule = read_xyz_content(
        """3

        O 0 0 0
        H 1 0 0
        F -0.8 0.2 0
        """.split("\n")
    )
    basis_set = read_basis("3-21g")
    attach_basis_to_molecule(molecule, basis_set)

    reference_angular_list = [ 0,0,1,0,1, 0,0, 0,0,1,0,1 ]
    reference_i_atom_list = [ 0,0,0,0,0, 1,1, 2,2,2,2,2 ]
    reference_i_ao_start_list = [ 0,1,2,5,6, 9,10, 11,12,13,16,17 ]
    reference_n_ao = 20

    for i_shell, shell in enumerate(molecule.basis_shells):
        assert shell.angular == reference_angular_list[i_shell]
        assert shell.i_atom == reference_i_atom_list[i_shell]
        assert shell.i_ao_start == reference_i_ao_start_list[i_shell]
    assert molecule.n_ao == reference_n_ao

    assign_ao_index_to_shell(molecule, AtomicOrbitalOrder.ANGULAR_LEADING)

    reference_angular_list = [ 0,0,0, 0,0, 0,0,0, 1,1, 1,1 ]
    reference_i_atom_list = [ 0,0,0, 1,1, 2,2,2, 0,0, 2,2 ]
    reference_i_ao_start_list = [ 0,1,2, 3,4, 5,6,7, 8,11, 14,17 ]
    reference_n_ao = 20

    for i_shell, shell in enumerate(molecule.basis_shells):
        assert shell.angular == reference_angular_list[i_shell]
        assert shell.i_atom == reference_i_atom_list[i_shell]
        assert shell.i_ao_start == reference_i_ao_start_list[i_shell]
    assert molecule.n_ao == reference_n_ao

def test_basis_assignment_high_angular():
    molecule = read_xyz_content(
        """3

        O 0 0 0
        H 1 0 0
        H -0.8 0.2 0
        """.split("\n")
    )
    basis_set = read_basis("def2-tzvp")
    attach_basis_to_molecule(molecule, basis_set)

    reference_angular_list = [ 0,0,0,0,0,1,1,1,2,2,3, 0,0,0,1, 0,0,0,1 ]
    reference_i_atom_list = [ 0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1, 2,2,2,2 ]
    reference_i_ao_start_list = [ 0,1,2,3,4,5,8,11,14,19,24, 31,32,33,34, 37,38,39,40 ]
    reference_n_ao = 43

    for i_shell, shell in enumerate(molecule.basis_shells):
        assert shell.angular == reference_angular_list[i_shell]
        assert shell.i_atom == reference_i_atom_list[i_shell]
        assert shell.i_ao_start == reference_i_ao_start_list[i_shell]
    assert molecule.n_ao == reference_n_ao

    molecule.spherical_basis = False
    assign_ao_index_to_shell(molecule, AtomicOrbitalOrder.ANGULAR_LEADING)

    reference_angular_list = [ 0,0,0,0,0, 0,0,0, 0,0,0, 1,1,1, 1, 1, 2,2, 3 ]
    reference_i_atom_list = [ 0,0,0,0,0, 1,1,1, 2,2,2, 0,0,0, 1, 2, 0,0, 0 ]
    reference_i_ao_start_list = [ 0,1,2,3,4, 5,6,7, 8,9,10, 11,14,17, 20, 23, 26,32, 38 ]
    reference_n_ao = 48

    for i_shell, shell in enumerate(molecule.basis_shells):
        assert shell.angular == reference_angular_list[i_shell]
        assert shell.i_atom == reference_i_atom_list[i_shell]
        assert shell.i_ao_start == reference_i_ao_start_list[i_shell]
    assert molecule.n_ao == reference_n_ao
