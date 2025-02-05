
import pytest

import numpy as np

from electron_integral_playground.units import LengthUnits
from electron_integral_playground.geometry_reader.permissive_xyz_reader import read_xyz_content
from electron_integral_playground.basis_reader.gaussian_basis_reader import read_basis, read_basis_content
from electron_integral_playground.basis_utility import attach_basis_to_molecule
from electron_integral_playground.pair_utility import form_pair_data
from electron_integral_playground.molecular_integral.nuclear_attraction import nuclear_attraction

import pathlib
current_directory = pathlib.Path(__file__).parent.resolve()
reference_directory = current_directory / "reference_nuclear_attraction"

pyscf_bohr_integral_threshold = 1e-14
pyscf_angstrom_integral_threshold = 1e-9

def test_nuclear_attraction_all_angular_spherical_h2():
    molecule = read_xyz_content(
        """
        H 0 0 0
        H 1.0 1.3 1.55
        """,
        unit = LengthUnits.BOHR
    )
    molecule.spherical_basis = True
    basis_set = read_basis_content("""
        H     0
        S    2   1.00
            1.0    1.0
            0.5    1.0
        P    2   1.00
            1.2    1.0
            0.6    1.0
        D    2   1.00
            1.4    1.0
            0.7    1.0
        F    2   1.00
            1.6    1.0
            0.8    1.0
        G    2   1.00
            1.8    1.0
            0.9    1.0
        H    2   1.00
            2.0    1.0
            1.0    1.0
        I    2   1.00
            2.2    1.0
            1.1    1.0
        ****
        """)
    attach_basis_to_molecule(molecule, basis_set)
    pair_data = form_pair_data(molecule, 1e-14)

    charge_position = np.array([
        [ 0, 0, 0, ],
        [ 1.0, 1.3, 1.55, ],
        [ 0.5, 0.65, 0.775, ],
        [ 10, 0, 0, ],
        [ 10, 2, 0, ],
        [ 50, 2, 0, ],
        [ 100, 2, 0, ],
        [ 500, 2, 0, ],
        [ 1, 0, 0, ],
        [ 0, 1, 0, ],
        [ 0, 0, 1, ],
        [ 1, 1, 1, ],
        [ 0, 0, 0.001, ],
        [ -3, 0, 0, ],
    ])
    test_V = nuclear_attraction(pair_data, molecule, charge_position)

    ref_V = np.loadtxt(reference_directory / 'reference_nuclear_attraction_all_angular_spherical_h2_data.txt')
    ref_V = ref_V.reshape(test_V.shape)

    np.testing.assert_allclose(test_V, ref_V, atol = pyscf_bohr_integral_threshold)

def test_nuclear_attraction_all_angular_cartesian_h2():
    molecule = read_xyz_content(
        """
        H 0 0 0
        H 1.0 1.3 1.55
        """,
        unit = LengthUnits.BOHR
    )
    molecule.spherical_basis = False
    basis_set = read_basis_content("""
        H     0
        S    2   1.00
            1.0    1.0
            0.5    1.0
        P    2   1.00
            1.2    1.0
            0.6    1.0
        D    2   1.00
            1.4    1.0
            0.7    1.0
        F    2   1.00
            1.6    1.0
            0.8    1.0
        G    2   1.00
            1.8    1.0
            0.9    1.0
        H    2   1.00
            2.0    1.0
            1.0    1.0
        I    2   1.00
            2.2    1.0
            1.1    1.0
        ****
        """)
    attach_basis_to_molecule(molecule, basis_set)
    pair_data = form_pair_data(molecule, 1e-14)

    charge_position = np.array([
        [ 0, 0, 0, ],
        [ 1.0, 1.3, 1.55, ],
        [ 0.5, 0.65, 0.775, ],
        [ 10, 0, 0, ],
        [ 10, 2, 0, ],
        [ 50, 2, 0, ],
        [ 100, 2, 0, ],
        [ 500, 2, 0, ],
        [ 1, 0, 0, ],
        [ 0, 1, 0, ],
        [ 0, 0, 1, ],
        [ 1, 1, 1, ],
        [ 0, 0, 0.001, ],
        [ -3, 0, 0, ],
    ])
    test_V = nuclear_attraction(pair_data, molecule, charge_position)

    ref_V = np.loadtxt(reference_directory / 'reference_nuclear_attraction_all_angular_cartesian_h2_data.txt')
    ref_V = ref_V.reshape(test_V.shape)

    np.testing.assert_allclose(test_V, ref_V, atol = pyscf_bohr_integral_threshold)

def test_nuclear_attraction_hof():
    molecule = read_xyz_content(
        """
        H -1 0.1 0
        O 0 0 0
        F 0.5 0.6 0.7
        """
    )
    basis_set = read_basis("def2-svp")
    attach_basis_to_molecule(molecule, basis_set)
    pair_data = form_pair_data(molecule, 1e-14)

    nuclear_position = molecule.geometry
    test_V = nuclear_attraction(pair_data, molecule, nuclear_position)

    ref_V = np.loadtxt(reference_directory / 'reference_nuclear_attraction_hof_data.txt')
    ref_V = ref_V.reshape(test_V.shape)

    np.testing.assert_allclose(test_V, ref_V, atol = pyscf_angstrom_integral_threshold)

def test_nuclear_attraction_hof_distorted():
    molecule = read_xyz_content(
        """
        H -1 0.1 0
        O 0 0 0
        F 0.5 0.6 2.7
        """
    )
    basis_set = read_basis("def2-svp")
    attach_basis_to_molecule(molecule, basis_set)
    pair_data = form_pair_data(molecule, 1e-14)

    nuclear_position = molecule.geometry
    test_V = nuclear_attraction(pair_data, molecule, nuclear_position)

    ref_V = np.loadtxt(reference_directory / 'reference_nuclear_attraction_hof_distorted_data.txt')
    ref_V = ref_V.reshape(test_V.shape)

    np.testing.assert_allclose(test_V, ref_V, atol = pyscf_angstrom_integral_threshold)

def test_nuclear_attraction_all_angular_spherical_h2_omega():
    molecule = read_xyz_content(
        """
        H 0 0 0
        H 1.0 1.3 1.55
        """,
        unit = LengthUnits.BOHR
    )
    molecule.spherical_basis = True
    basis_set = read_basis_content("""
        H     0
        S    2   1.00
            1.0    1.0
            0.5    1.0
        P    2   1.00
            1.2    1.0
            0.6    1.0
        D    2   1.00
            1.4    1.0
            0.7    1.0
        F    2   1.00
            1.6    1.0
            0.8    1.0
        G    2   1.00
            1.8    1.0
            0.9    1.0
        H    2   1.00
            2.0    1.0
            1.0    1.0
        I    2   1.00
            2.2    1.0
            1.1    1.0
        ****
        """)
    attach_basis_to_molecule(molecule, basis_set)
    pair_data = form_pair_data(molecule, 1e-14)

    charge_position = np.array([
        [ 0, 0, 0, ],
        [ 1.0, 1.3, 1.55, ],
        [ 0.5, 0.65, 0.775, ],
        [ 10, 0, 0, ],
        [ 10, 2, 0, ],
        [ 50, 2, 0, ],
        [ 100, 2, 0, ],
        [ 500, 2, 0, ],
        [ 1, 0, 0, ],
        [ 0, 1, 0, ],
        [ 0, 0, 1, ],
        [ 1, 1, 1, ],
        [ 0, 0, 0.001, ],
        [ -3, 0, 0, ],
    ])
    omega = 0.3
    test_V = nuclear_attraction(pair_data, molecule, charge_position, omega)

    ref_V = np.loadtxt(reference_directory / 'reference_nuclear_attraction_all_angular_spherical_h2_omega_data.txt')
    ref_V = ref_V.reshape(test_V.shape)

    np.testing.assert_allclose(test_V, ref_V, atol = pyscf_bohr_integral_threshold)

def test_nuclear_attraction_all_angular_cartesian_h2_omega():
    molecule = read_xyz_content(
        """
        H 0 0 0
        H 1.0 1.3 1.55
        """,
        unit = LengthUnits.BOHR
    )
    molecule.spherical_basis = False
    basis_set = read_basis_content("""
        H     0
        S    2   1.00
            1.0    1.0
            0.5    1.0
        P    2   1.00
            1.2    1.0
            0.6    1.0
        D    2   1.00
            1.4    1.0
            0.7    1.0
        F    2   1.00
            1.6    1.0
            0.8    1.0
        G    2   1.00
            1.8    1.0
            0.9    1.0
        H    2   1.00
            2.0    1.0
            1.0    1.0
        I    2   1.00
            2.2    1.0
            1.1    1.0
        ****
        """)
    attach_basis_to_molecule(molecule, basis_set)
    pair_data = form_pair_data(molecule, 1e-14)

    charge_position = np.array([
        [ 0, 0, 0, ],
        [ 1.0, 1.3, 1.55, ],
        [ 0.5, 0.65, 0.775, ],
        [ 10, 0, 0, ],
        [ 10, 2, 0, ],
        [ 50, 2, 0, ],
        [ 100, 2, 0, ],
        [ 500, 2, 0, ],
        [ 1, 0, 0, ],
        [ 0, 1, 0, ],
        [ 0, 0, 1, ],
        [ 1, 1, 1, ],
        [ 0, 0, 0.001, ],
        [ -3, 0, 0, ],
    ])
    omega = 0.3
    test_V = nuclear_attraction(pair_data, molecule, charge_position, omega)

    ref_V = np.loadtxt(reference_directory / 'reference_nuclear_attraction_all_angular_cartesian_h2_omega_data.txt')
    ref_V = ref_V.reshape(test_V.shape)

    np.testing.assert_allclose(test_V, ref_V, atol = pyscf_bohr_integral_threshold)

def test_nuclear_attraction_hof_omega():
    molecule = read_xyz_content(
        """
        H -1 0.1 0
        O 0 0 0
        F 0.5 0.6 0.7
        """
    )
    basis_set = read_basis("def2-svp")
    attach_basis_to_molecule(molecule, basis_set)
    pair_data = form_pair_data(molecule, 1e-14)

    nuclear_position = molecule.geometry
    omega = 0.3
    test_V = nuclear_attraction(pair_data, molecule, nuclear_position, omega)

    ref_V = np.loadtxt(reference_directory / 'reference_nuclear_attraction_hof_omega_data.txt')
    ref_V = ref_V.reshape(test_V.shape)

    np.testing.assert_allclose(test_V, ref_V, atol = pyscf_angstrom_integral_threshold)

def test_nuclear_attraction_hof_distorted_omega():
    molecule = read_xyz_content(
        """
        H -1 0.1 0
        O 0 0 0
        F 0.5 0.6 2.7
        """
    )
    basis_set = read_basis("def2-svp")
    attach_basis_to_molecule(molecule, basis_set)
    pair_data = form_pair_data(molecule, 1e-14)

    nuclear_position = molecule.geometry
    omega = 0.3
    test_V = nuclear_attraction(pair_data, molecule, nuclear_position, omega)

    ref_V = np.loadtxt(reference_directory / 'reference_nuclear_attraction_hof_distorted_omega_data.txt')
    ref_V = ref_V.reshape(test_V.shape)

    np.testing.assert_allclose(test_V, ref_V, atol = pyscf_angstrom_integral_threshold)
