
import pytest

import numpy as np

from electron_integral_playground.units import LengthUnits
from electron_integral_playground.geometry_reader.xyz_reader import read_xyz_content
from electron_integral_playground.basis_reader.gaussian_basis_reader import read_basis, read_basis_content
from electron_integral_playground.basis_utility import attach_auxiliary_basis_to_molecule
from electron_integral_playground.pair_utility import form_auxiliary_pair_data
from electron_integral_playground.molecular_integral.two_center import two_center

import pathlib
current_directory = pathlib.Path(__file__).parent.resolve()
reference_directory = current_directory / "reference_two_center"

pyscf_bohr_integral_threshold = 1e-12
pyscf_angstrom_integral_threshold = 1e-9

def test_two_center_all_angular_spherical_h2():
    molecule = read_xyz_content(
        """2

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
        K    2   1.00
            1.9    1.0
            0.9    1.0
        L    2   1.00
            1.8    1.0
            0.8    1.0
        M    2   1.00
            1.7    1.0
            0.7    1.0
        ****
        """)
    attach_auxiliary_basis_to_molecule(molecule, basis_set)
    pair_data = form_auxiliary_pair_data(molecule)
    test_J2c = two_center(pair_data, molecule)

    ref_J2c = np.loadtxt(reference_directory / 'reference_two_center_all_angular_spherical_h2_data.txt')

    np.testing.assert_allclose(test_J2c, ref_J2c, atol = pyscf_bohr_integral_threshold)

def test_two_center_all_angular_cartesian_h2():
    molecule = read_xyz_content(
        """2

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
        K    2   1.00
            1.9    1.0
            0.9    1.0
        L    2   1.00
            1.8    1.0
            0.8    1.0
        M    2   1.00
            1.7    1.0
            0.7    1.0
        ****
        """)
    attach_auxiliary_basis_to_molecule(molecule, basis_set)
    pair_data = form_auxiliary_pair_data(molecule)
    test_J2c = two_center(pair_data, molecule)

    ref_J2c = np.loadtxt(reference_directory / 'reference_two_center_all_angular_cartesian_h2_data.txt')

    np.testing.assert_allclose(test_J2c, ref_J2c, atol = pyscf_bohr_integral_threshold)

def test_two_center_hof():
    molecule = read_xyz_content(
        """3

        H -1 0.1 0
        O 0 0 0
        F 0.5 0.6 0.7
        """
    )
    basis_set = read_basis("def2-svp-rifit")
    attach_auxiliary_basis_to_molecule(molecule, basis_set)
    pair_data = form_auxiliary_pair_data(molecule)
    test_J2c = two_center(pair_data, molecule)

    ref_J2c = np.loadtxt(reference_directory / 'reference_two_center_hof_data.txt')

    np.testing.assert_allclose(test_J2c, ref_J2c, atol = pyscf_angstrom_integral_threshold)

def test_two_center_hof_distorted():
    molecule = read_xyz_content(
        """3

        H -1 0.1 0
        O 0 0 0
        F 0.5 0.6 2.7
        """
    )
    basis_set = read_basis("def2-svp-rifit")
    attach_auxiliary_basis_to_molecule(molecule, basis_set)
    pair_data = form_auxiliary_pair_data(molecule)
    test_J2c = two_center(pair_data, molecule)

    ref_J2c = np.loadtxt(reference_directory / 'reference_two_center_hof_distorted_data.txt')

    np.testing.assert_allclose(test_J2c, ref_J2c, atol = pyscf_angstrom_integral_threshold)

def test_two_center_all_angular_spherical_h2_omega():
    molecule = read_xyz_content(
        """2

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
        K    2   1.00
            1.9    1.0
            0.9    1.0
        L    2   1.00
            1.8    1.0
            0.8    1.0
        M    2   1.00
            1.7    1.0
            0.7    1.0
        ****
        """)
    attach_auxiliary_basis_to_molecule(molecule, basis_set)
    pair_data = form_auxiliary_pair_data(molecule)
    omega = 0.3
    test_J2c = two_center(pair_data, molecule, omega)

    ref_J2c = np.loadtxt(reference_directory / 'reference_two_center_all_angular_spherical_h2_omega_data.txt')

    np.testing.assert_allclose(test_J2c, ref_J2c, atol = pyscf_bohr_integral_threshold)

def test_two_center_all_angular_cartesian_h2_omega():
    molecule = read_xyz_content(
        """2

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
        K    2   1.00
            1.9    1.0
            0.9    1.0
        L    2   1.00
            1.8    1.0
            0.8    1.0
        M    2   1.00
            1.7    1.0
            0.7    1.0
        ****
        """)
    attach_auxiliary_basis_to_molecule(molecule, basis_set)
    pair_data = form_auxiliary_pair_data(molecule)
    omega = 0.3
    test_J2c = two_center(pair_data, molecule, omega)

    ref_J2c = np.loadtxt(reference_directory / 'reference_two_center_all_angular_cartesian_h2_omega_data.txt')

    np.testing.assert_allclose(test_J2c, ref_J2c, atol = pyscf_bohr_integral_threshold)

def test_two_center_hof_omega():
    molecule = read_xyz_content(
        """3

        H -1 0.1 0
        O 0 0 0
        F 0.5 0.6 0.7
        """
    )
    basis_set = read_basis("def2-svp-rifit")
    attach_auxiliary_basis_to_molecule(molecule, basis_set)
    pair_data = form_auxiliary_pair_data(molecule)
    omega = 0.3
    test_J2c = two_center(pair_data, molecule, omega)

    ref_J2c = np.loadtxt(reference_directory / 'reference_two_center_hof_omega_data.txt')

    np.testing.assert_allclose(test_J2c, ref_J2c, atol = pyscf_angstrom_integral_threshold)

def test_two_center_hof_distorted_omega():
    molecule = read_xyz_content(
        """3

        H -1 0.1 0
        O 0 0 0
        F 0.5 0.6 2.7
        """
    )
    basis_set = read_basis("def2-svp-rifit")
    attach_auxiliary_basis_to_molecule(molecule, basis_set)
    pair_data = form_auxiliary_pair_data(molecule)
    omega = 0.3
    test_J2c = two_center(pair_data, molecule, omega)

    ref_J2c = np.loadtxt(reference_directory / 'reference_two_center_hof_distorted_omega_data.txt')

    np.testing.assert_allclose(test_J2c, ref_J2c, atol = pyscf_angstrom_integral_threshold)
