
import pytest

import numpy as np

from electron_integral_playground.data import molecules_path
from electron_integral_playground.geometry_reader.xyz_reader import read_xyz_file, read_xyz_content
from electron_integral_playground.units import angstrom_to_bohr

xyz_read_threshold = 1e-16

def test_read_xyz_file():
    xyz_filename = molecules_path / "h2o2.xyz"
    molecule = read_xyz_file(xyz_filename)
    reference_elements = [ "O", "O", "H", "H" ]
    assert molecule.elements == reference_elements
    assert molecule.n_ao <= 0
    assert molecule.spherical_basis
    reference_geometry = np.array([[ 0.0000,  0.7375, -0.0528],
                                   [ 0.0000, -0.7375, -0.0528],
                                   [ 0.8190,  0.8170,  0.4220],
                                   [-0.8190, -0.8170,  0.4220],]) * angstrom_to_bohr
    np.testing.assert_allclose(molecule.geometry, reference_geometry, atol = xyz_read_threshold)

def test_read_xyz_content():
    xyz = """3

    O 0 0 1
    H 1.2 0 1
    I -0.8 0.2 1
    """
    molecule = read_xyz_content(xyz)
    reference_elements = [ "O", "H", "I" ]
    assert molecule.elements == reference_elements
    assert molecule.n_ao <= 0
    assert molecule.spherical_basis
    reference_geometry = np.array([[   0,   0,   1, ],
                                   [ 1.2,   0,   1, ],
                                   [-0.8, 0.2,   1, ],]) * angstrom_to_bohr
    np.testing.assert_allclose(molecule.geometry, reference_geometry, atol = xyz_read_threshold)
