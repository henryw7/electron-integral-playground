
import pytest

import numpy as np

from electron_integral_playground.units import LengthUnits
from electron_integral_playground.geometry_reader.xyz_reader import read_xyz_content
from electron_integral_playground.basis_reader.gaussian_basis_reader import read_basis, read_basis_content
from electron_integral_playground.basis_utility import attach_basis_to_molecule
from electron_integral_playground.pair_utility import form_pair_data
from electron_integral_playground.molecular_integral.overlap import overlap

import pathlib
current_directory = pathlib.Path(__file__).parent.resolve()
reference_directory = current_directory / "reference_overlap"

pyscf_bohr_integral_threshold = 1e-14
pyscf_angstrom_integral_threshold = 1e-10

def test_overlap_all_angular_spherical_h2():
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
        ****
        """)
    attach_basis_to_molecule(molecule, basis_set)
    pair_data = form_pair_data(molecule, 1e-14)
    test_S = overlap(pair_data, molecule)

    ref_S = np.loadtxt(reference_directory / 'reference_overlap_all_angular_spherical_h2_data.txt')

    np.testing.assert_allclose(test_S, ref_S, atol = pyscf_bohr_integral_threshold)

def test_overlap_all_angular_cartesian_h2():
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
        ****
        """)
    attach_basis_to_molecule(molecule, basis_set)
    pair_data = form_pair_data(molecule, 1e-14)
    test_S = overlap(pair_data, molecule)

    ref_S = np.loadtxt(reference_directory / 'reference_overlap_all_angular_cartesian_h2_data.txt')

    np.testing.assert_allclose(test_S, ref_S, atol = pyscf_bohr_integral_threshold)

def test_overlap_hof():
    molecule = read_xyz_content(
        """3

        H -1 0.1 0
        O 0 0 0
        F 0.5 0.6 0.7
        """
    )
    basis_set = read_basis("def2-svp")
    attach_basis_to_molecule(molecule, basis_set)
    pair_data = form_pair_data(molecule, 1e-14)
    test_S = overlap(pair_data, molecule)

    ref_S = np.loadtxt(reference_directory / 'reference_overlap_hof_data.txt')

    np.testing.assert_allclose(test_S, ref_S, atol = pyscf_angstrom_integral_threshold)

def test_overlap_hof_distorted():
    molecule = read_xyz_content(
        """3

        H -1 0.1 0
        O 0 0 0
        F 0.5 0.6 2.7
        """
    )
    basis_set = read_basis("def2-svp")
    attach_basis_to_molecule(molecule, basis_set)
    pair_data = form_pair_data(molecule, 1e-14)
    test_S = overlap(pair_data, molecule)

    ref_S = np.loadtxt(reference_directory / 'reference_overlap_hof_distorted_data.txt')

    np.testing.assert_allclose(test_S, ref_S, atol = pyscf_angstrom_integral_threshold)
