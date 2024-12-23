
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

type BasisSet = dict[str, list[GaussianShell]]

@dataclass
class Molecule:
    elements: list[str]
    geometry: np.ndarray # Bohr
    basis_shells: list[GaussianShell]
    n_ao: int

@dataclass
class PrimitivePair:
    i_shell: int
    j_shell: int
    i_primitive: int
    j_primitive: int
    upper_bound: float

type PrimitivePairList = map[tuple[int, int], list[PrimitivePair]]

@dataclass
class PrimitivePairData:
    P_p: np.ndarray
    A_a: np.ndarray
    B_b: np.ndarray
    coefficient: np.ndarray
    i_ao_start: np.ndarray
    j_ao_start: np.ndarray
    i_atom: np.ndarray
    j_atom: np.ndarray
    ij_spherical: np.ndarray

type PrimitivePairDataAngularList = map[tuple[int, int], list[PrimitivePairData]]
