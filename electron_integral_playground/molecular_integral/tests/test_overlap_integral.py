
import pytest

import numpy as np

from electron_integral_playground.units import LengthUnits
from electron_integral_playground.geometry_reader.xyz_reader import read_xyz_content
from electron_integral_playground.basis_reader.gaussian_basis_reader import read_basis, read_basis_content
from electron_integral_playground.basis_utility import attach_basis_to_molecule
from electron_integral_playground.pair_utility import form_primitive_pair_list, form_primitive_pair_data
from electron_integral_playground.molecular_integral.overlap import overlap

integral_threshold = 1e-14

def test_overlap_all_angular_spherical_h2():
    molecule = read_xyz_content(
        """2

        H 0 0 0
        H 1.0 1.3 1.55
        """.split("\n"),
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
        """.split("\n"))
    attach_basis_to_molecule(molecule, basis_set)
    pair_list = form_primitive_pair_list(molecule, 1e-14)
    pair_data_list = form_primitive_pair_data(molecule, pair_list)
    test_S = overlap(pair_data_list, molecule)

    ref_S = np.loadtxt('reference_overlap_all_angular_spherical_h2_data.txt')

    np.testing.assert_allclose(test_S, ref_S, atol = integral_threshold)
