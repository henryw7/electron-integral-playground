
import numpy as np

from electron_integral_playground.data_structure import Molecule
from electron_integral_playground.units import LengthUnits, angstrom_to_bohr

def read_xyz_content(xyz_file_lines: list[str], unit: LengthUnits = LengthUnits.ANGSTROM) -> Molecule:
    assert len(xyz_file_lines) >= 2
    n_atom = int(xyz_file_lines[0])
    assert n_atom > 0
    assert len(xyz_file_lines) >= n_atom + 2

    atom_types = [None] * n_atom
    atom_coordinates = np.empty((n_atom, 3))
    for i_atom in range(n_atom):
        line = xyz_file_lines[i_atom + 2]
        fields = line.split()
        assert len(fields) == 4
        atom_types[i_atom] = fields[0]
        atom_coordinates[i_atom, :] = [float(fields[1]), float(fields[2]), float(fields[3])]

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
        n_ao = -1,
    )

def read_xyz_file(xyz_filename: str, unit: LengthUnits = LengthUnits.ANGSTROM) -> Molecule:
    xyz_file = open(xyz_filename)
    xyz_file_lines = xyz_file.readlines()
    xyz_file.close()
    return read_xyz_content(xyz_file_lines, unit)
