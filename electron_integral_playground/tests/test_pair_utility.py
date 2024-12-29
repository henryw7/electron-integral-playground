
import pytest

import numpy as np

from electron_integral_playground.geometry_reader.xyz_reader import read_xyz_content
from electron_integral_playground.basis_reader.gaussian_basis_reader import read_basis
from electron_integral_playground.basis_utility import attach_basis_to_molecule
from electron_integral_playground.pair_utility import form_primitive_pair_list, form_primitive_pair_data

def test_pair_list_formation_compact():
    molecule = read_xyz_content(
        """3

        O 0 0 0
        H 1 0 0
        F -0.8 0.2 0
        """.split("\n")
    )
    basis_set = read_basis("def2-svp")
    attach_basis_to_molecule(molecule, basis_set)
    test_pair_list = form_primitive_pair_list(molecule, 1e-14)

    ref_n_pair = { (0,0) : 171, (0,1) : 158, (0,2): 36, (1,1) : 51, (1,2) : 18, (2,2) : 3}

    assert len(test_pair_list) == len(ref_n_pair)
    for ij_angular in test_pair_list:
        assert len(test_pair_list[ij_angular]) == ref_n_pair[ij_angular]
        for i in range(len(test_pair_list[ij_angular]) - 1):
            assert test_pair_list[ij_angular][i].upper_bound >= test_pair_list[ij_angular][i + 1].upper_bound
            assert test_pair_list[ij_angular][i].upper_bound >= 1e-14

def test_pair_list_formation_sparse():
    molecule = read_xyz_content(
        """3

        O 0 0 0
        H 3 0 0
        F -0.8 2.2 0
        """.split("\n")
    )
    basis_set = read_basis("def2-svp")
    attach_basis_to_molecule(molecule, basis_set)
    test_pair_list = form_primitive_pair_list(molecule, 1e-14)

    ref_n_pair = { (0,0) : 144, (0,1) : 142, (0,2): 34, (1,1) : 47, (1,2) : 18, (2,2) : 3}

    assert len(test_pair_list) == len(ref_n_pair)
    for ij_angular in test_pair_list:
        assert len(test_pair_list[ij_angular]) == ref_n_pair[ij_angular]
        for i in range(len(test_pair_list[ij_angular]) - 1):
            assert test_pair_list[ij_angular][i].upper_bound >= test_pair_list[ij_angular][i + 1].upper_bound
            assert test_pair_list[ij_angular][i].upper_bound >= 1e-14

def test_pair_data_formation():
    molecule = read_xyz_content(
        """3

        O 0 0 0
        H 1 0 0
        F -0.8 0.2 0
        """.split("\n")
    )
    basis_set = read_basis("def2-svp")
    attach_basis_to_molecule(molecule, basis_set)
    pair_list = form_primitive_pair_list(molecule, 1e-14)
    test_pair_data_list = form_primitive_pair_data(molecule, pair_list)

    ref_n_pair = { (0,0) : 171, (0,1) : 158, (0,2): 36, (1,1) : 51, (1,2) : 18, (2,2) : 3}

    assert len(test_pair_data_list) == len(ref_n_pair)
    for ij_angular in test_pair_data_list:
        assert test_pair_data_list[ij_angular].P_p.shape == (ref_n_pair[ij_angular], 4)
        assert test_pair_data_list[ij_angular].A_a.shape == (ref_n_pair[ij_angular], 4)
        assert test_pair_data_list[ij_angular].B_b.shape == (ref_n_pair[ij_angular], 4)
        assert test_pair_data_list[ij_angular].coefficient.shape == (ref_n_pair[ij_angular], )
        assert test_pair_data_list[ij_angular].i_ao_start.shape == (ref_n_pair[ij_angular], )
        assert test_pair_data_list[ij_angular].j_ao_start.shape == (ref_n_pair[ij_angular], )
        assert test_pair_data_list[ij_angular].i_atom.shape == (ref_n_pair[ij_angular], )
        assert test_pair_data_list[ij_angular].j_atom.shape == (ref_n_pair[ij_angular], )
        assert test_pair_data_list[ij_angular].P_p.dtype == np.float64
        assert test_pair_data_list[ij_angular].A_a.dtype == np.float64
        assert test_pair_data_list[ij_angular].B_b.dtype == np.float64
        assert test_pair_data_list[ij_angular].coefficient.dtype == np.float64
        assert test_pair_data_list[ij_angular].i_ao_start.dtype == np.int32
        assert test_pair_data_list[ij_angular].j_ao_start.dtype == np.int32
        assert test_pair_data_list[ij_angular].i_atom.dtype == np.int32
        assert test_pair_data_list[ij_angular].j_atom.dtype == np.int32
        assert any(test_pair_data_list[ij_angular].i_ao_start >= 0)
        assert any(test_pair_data_list[ij_angular].j_ao_start >= 0)
        assert any(test_pair_data_list[ij_angular].i_atom >= 0)
        assert any(test_pair_data_list[ij_angular].j_atom >= 0)
