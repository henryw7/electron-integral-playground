
import pytest
import pathlib
CWD = pathlib.Path(__file__).parent.resolve()

import numpy as np

from electron_integral_playground.geometry_reader.xyz_reader import read_xyz_file
from electron_integral_playground.geometry import Molecule
from electron_integral_playground.units import angstrom_to_bohr

def test_read_xyz_file():
    xyz_filename = CWD / "h2o2.xyz"
    molecule = read_xyz_file(xyz_filename)
    assert type(molecule) == Molecule
    reference_elements = [ "O", "O", "H", "H" ]
    assert molecule.elements == reference_elements
    reference_geometry = np.array([[ 0.0000,  0.7375, -0.0528],
                                   [ 0.0000, -0.7375, -0.0528],
                                   [ 0.8190,  0.8170,  0.4220],
                                   [-0.8190, -0.8170,  0.4220],]) * angstrom_to_bohr
    np.testing.assert_allclose(molecule.geometry, reference_geometry, atol = 1e-14)
