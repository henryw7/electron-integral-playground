
from dataclasses import dataclass

import numpy as np

@dataclass
class GaussianShell:
    angular: int
    spherical: bool
    i_ao_start: int
    i_atom: int
    primitive_exponents: np.ndarray
    primitive_coefficients: np.ndarray

@dataclass
class Molecule:
    elements: list[str]
    geometry: np.ndarray # Bohr
    basis_shells: list[GaussianShell]
