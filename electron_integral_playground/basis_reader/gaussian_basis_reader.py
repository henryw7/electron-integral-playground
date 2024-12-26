
from pathlib import Path
import numpy as np

from electron_integral_playground.basis_utility import angular_letter_to_l_value
from electron_integral_playground.data_structure import GaussianShell, BasisSet
from electron_integral_playground.data import basis_sets_path

def read_basis_content(basis_file_lines: list[str]) -> BasisSet:
    basis_set = {}
    current_atom = None
    current_orbitals = None
    n_primitive = 0
    primitive_list = None
    angular_letter = None
    for line in basis_file_lines:
        if "ECP" in line:
            raise NotImplementedError("ECP not supported")
        fields = line.split()
        if not fields:
            continue # Empty line
        if fields[0].startswith("!"):
            continue # Comment
        if fields[0].startswith("****"):
            assert current_atom is not None
            assert current_orbitals is not None
            basis_set[current_atom] = current_orbitals
            current_atom = None
            current_orbitals = None
            continue
        if current_orbitals is None:
            assert len(fields) == 2 and fields[1] == "0"
            current_atom = fields[0]
            assert current_atom not in basis_set
            current_orbitals = []
            continue
        assert current_orbitals is not None
        if n_primitive <= 0:
            assert len(fields) == 3 and float(fields[2]) == 1.0
            angular_letter = fields[0]
            assert len(angular_letter) == 1 or angular_letter == "SP"
            n_primitive = int(fields[1])
            assert n_primitive > 0
            primitive_list = []
        else:
            if len(angular_letter) == 1:
                assert len(fields) == 2
                exponent = float(fields[0].replace("D", "E"))
                assert exponent > 0.0
                coefficient = float(fields[1].replace("D", "E"))
                primitive_list.append([exponent, coefficient])
            else:
                assert angular_letter == "SP"
                exponent = float(fields[0].replace("D", "E"))
                assert exponent > 0.0
                coefficient_S = float(fields[1].replace("D", "E"))
                coefficient_P = float(fields[2].replace("D", "E"))
                primitive_list.append([exponent, coefficient_S, coefficient_P])

            n_primitive -= 1
            if n_primitive <= 0:
                primitive_list = np.array(primitive_list)
                if len(angular_letter) == 1:
                    current_orbitals.append(GaussianShell(
                        angular = angular_letter_to_l_value(angular_letter),
                        i_ao_start = -1,
                        i_atom = -1,
                        primitive_exponents = primitive_list[:, 0],
                        primitive_coefficients = primitive_list[:, 1],
                    ))
                else:
                    assert angular_letter == "SP"
                    current_orbitals.append(GaussianShell(
                        angular = 0,
                        i_ao_start = -1,
                        i_atom = -1,
                        primitive_exponents = primitive_list[:, 0],
                        primitive_coefficients = primitive_list[:, 1],
                    ))
                    current_orbitals.append(GaussianShell(
                        angular = 1,
                        i_ao_start = -1,
                        i_atom = -1,
                        primitive_exponents = primitive_list[:, 0],
                        primitive_coefficients = primitive_list[:, 2],
                    ))
                primitive_list = None

    return basis_set

def read_basis_file(basis_filename: str) -> BasisSet:
    basis_file = open(basis_filename)
    basis_file_lines = basis_file.readlines()
    basis_file.close()
    return read_basis_content(basis_file_lines)

def read_basis(basis_name: str) -> BasisSet:
    basis_filename = basis_name.lower() + ".gaussian.basis"
    basis_filepath = basis_sets_path / basis_filename
    assert Path(basis_filepath).is_file(), f"Basis {basis_name} not in the basis set database {basis_sets_path}"
    return read_basis_file(basis_filepath)
