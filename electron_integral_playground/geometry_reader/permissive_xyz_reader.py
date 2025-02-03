
from typing import Union

import numpy as np

from electron_integral_playground.data_structure import Molecule
from electron_integral_playground.units import LengthUnits, angstrom_to_bohr

def read_xyz_content(xyz_lines: Union[str, list[str]], unit: LengthUnits = LengthUnits.ANGSTROM) -> Molecule:
    if type(xyz_lines) is str:
        xyz_lines = xyz_lines.split("\n")

    atom_types = []
    atom_coordinates = []
    for line in xyz_lines:
        fields = line.split()
        if (len(fields) == 0):
            continue
        assert len(fields) == 4
        atom_types.append(fields[0])
        atom_coordinates.append([float(fields[1]), float(fields[2]), float(fields[3])])

    assert len(atom_types) > 0
    atom_coordinates = np.array(atom_coordinates)

    if (unit == LengthUnits.BOHR):
        pass
    elif (unit == LengthUnits.ANGSTROM):
        atom_coordinates *= angstrom_to_bohr
    else:
        raise NotImplementedError(f"Unsupported unit {unit}")

    return Molecule(
        elements = atom_types,
        geometry = atom_coordinates,
        basis_shells = None,
        n_ao = 0,
        auxiliary_basis_shells = None,
        n_aux = 0,
        spherical_basis = True,
    )
