
from dataclasses import dataclass

import numpy as np

@dataclass
class GaussianShell:
    angular: int
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
    auxiliary_basis_shells: list[GaussianShell]
    n_aux: int
    spherical_basis: bool

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

type PrimitivePairDataAngularList = map[tuple[int, int], PrimitivePairData]

@dataclass
class Cell:
    dimension: int
    lattice_vectors: np.ndarray # Bohr, where each row (each 3 continuous values in a 3D lattice) stores a vector

@dataclass
class Crystal:
    cell: Cell
    atoms: Molecule # geometry field stores the absolute geometry of an arbitrary image of each atom

@dataclass
class PeriodicPrimitivePair:
    i_shell: int
    j_shell: int
    i_primitive: int
    j_primitive: int
    cell_offset: np.ndarray # of 3 integers in a 3D lattice
    upper_bound: float
