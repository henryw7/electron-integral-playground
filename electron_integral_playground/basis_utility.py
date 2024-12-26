
from enum import Enum

import numpy as np
from math import factorial
from copy import deepcopy

from electron_integral_playground.data_structure import GaussianShell, Molecule, BasisSet

angular_letter_to_l_value_map = { "S":0, "P":1, "D":2, "F":3, "G":4, "H":5, "I":6 }
def angular_letter_to_l_value(letter: str) -> int:
    assert len(letter) == 1
    letter = letter.upper()
    assert letter in angular_letter_to_l_value_map
    return angular_letter_to_l_value_map[letter]

def n_ao_of_angular(l: int, spherical: bool = True) -> int:
    if spherical:
        return 2 * l + 1
    else:
        return (l + 1) * (l + 2) // 2

def attach_basis_to_molecule(molecule: Molecule, basis_set: BasisSet) -> int:
    basis_set_actually_used = {}
    for element in molecule.elements:
        assert element in basis_set
        if element not in basis_set_actually_used:
            basis_set_actually_used[element] = basis_set[element]

    for element in basis_set_actually_used:
        for basis in basis_set_actually_used[element]:
            normalize_shell(basis)

    assert (molecule.basis_shells is None) or (not molecule.basis_shells)
    molecule.basis_shells = []
    for i_atom, element in enumerate(molecule.elements):
        basis_set = deepcopy(basis_set_actually_used[element])
        for basis in basis_set:
            basis.i_atom = i_atom
        molecule.basis_shells.extend(basis_set)

    return assign_ao_index_to_shell(molecule)

class AtomicOrbitalOrder(Enum):
    ATOM_LEADING = 1
    ANGULAR_LEADING = 2

def assign_ao_index_to_shell(molecule: Molecule, ao_order: AtomicOrbitalOrder = AtomicOrbitalOrder.ATOM_LEADING) -> None:
    assert len(molecule.basis_shells) > 0
    if ao_order == AtomicOrbitalOrder.ATOM_LEADING:
        molecule.basis_shells.sort(key = lambda shell: shell.i_atom)
    elif ao_order == AtomicOrbitalOrder.ANGULAR_LEADING:
        molecule.basis_shells.sort(key = lambda shell: shell.angular)
    else:
        raise NotImplementedError(f"Unsupported atomic orbital order {ao_order}")
    current_offset = 0
    for shell in molecule.basis_shells:
        shell.i_ao_start = current_offset
        current_offset += n_ao_of_angular(shell.angular, molecule.spherical_basis)
    molecule.n_ao = current_offset

def normalize_shell(shell: GaussianShell) -> None:
    """
    Modify the primitive_coefficients field of shell, to make sure the diagonal of the molecular overlap matrix is 1.
    For a non-S shell, the primitive are normalized according to the Cartesian pure x component (px, dxx, fxxx, gxxxx, etc.).
    This operation is NOT idempotent, don't call it more than once!
    """
    r"""
    For a contracted Gaussian function
    $$\phi(\vec{r}; \vec{i}, \vec{A}) = \sum_m^{n_{contract}} C_m^{contract} \chi(\vec{r}; \vec{i}, a_m, \vec{A})$$
    where $\chi(\vec{r})$ is an un-normalized primitive Gaussian function
    $$\chi(\vec{r}; \vec{i}, a, \vec{A}) = (x-A_x)^{i_x} (y-A_y)^{i_y} (z-A_z)^{i_z} e^{-a \left|\vec{r} - \vec{A}\right|^2}$$
    The normalization condition is
    $$\iiint_\infty d\vec{r} (\phi(\vec{r}))^2 = \sum_{m, n}^{n_{contract}} C_m^{contract} C_n^{contract} \iiint_\infty d\vec{r} \chi(\vec{r}; \vec{i}, a_m, \vec{A}) \chi(\vec{r}; \vec{i}, a_n, \vec{A}) = 1$$
    where the integral of two primtive Gaussian functions of the same angular index and the same atomic center is
    \begin{align*}
        \iiint_\infty d\vec{r} \chi(\vec{r}; \vec{i}, a_m, \vec{A}) \chi(\vec{r}; \vec{i}, a_n, \vec{A}) &= \iiint_\infty d\vec{r} (x-A_x)^{2i_x} (y-A_y)^{2i_y} (z-A_z)^{2i_z} e^{-(a_m + a_n) \left|\vec{r} - \vec{A}\right|^2} \\
            &= \iiint_\infty d\vec{r} \left( (x-A_x)^{i_x} (y-A_y)^{i_y} (z-A_z)^{i_z} e^{-\frac{a_m + a_n}{2} \left|\vec{r} - \vec{A}\right|^2} \right)^2 \\
            &= \iiint_\infty d\vec{r} \left( \chi\left(\vec{r}; \vec{i}, \frac{a_m + a_n}{2}, \vec{A}\right) \right)^2
    \end{align*}
    Given the normalization constant of an arbitrary primitive Gaussian function
    $$\chi^{normalized}(\vec{r}; \vec{i}, a, \vec{A}) = \left(\frac{2a}{\pi}\right)^{3/4} \left( \frac{(8a)^{i_x+i_y+i_z} i_x! i_y! i_z!}{(2i_x)! (2i_y)! (2i_z)!} \right)^{1/2} (x-A_x)^{i_x} (y-A_y)^{i_y} (z-A_z)^{i_z} e^{-a \left|\vec{r} - \vec{A}\right|^2}$$
    The integral above is
    $$\iiint_\infty d\vec{r} \left( \chi\left(\vec{r}; \vec{i}, \frac{a_m + a_n}{2}, \vec{A}\right) \right)^2 = \left(\frac{\pi}{a_m + a_n}\right)^{3/2} \frac{(2i_x)! (2i_y)! (2i_z)!}{4^{i_x+i_y+i_z} (a_m + a_n)^{i_x+i_y+i_z} i_x! i_y! i_z!}$$
    So the normalization condition simplifies to
    $$\iiint_\infty d\vec{r} (\phi(\vec{r}))^2 = \sum_{m, n}^{n_{contract}} C_m^{contract} C_n^{contract} \left(\frac{\pi}{a_m + a_n}\right)^{3/2} \frac{(2i_x)! (2i_y)! (2i_z)!}{4^{i_x+i_y+i_z} (a_m + a_n)^{i_x+i_y+i_z} i_x! i_y! i_z!} = 1$$
    Since $i_x + i_y + i_z = L$,
    $$\iiint_\infty d\vec{r} (\phi(\vec{r}))^2 = \sum_{m, n}^{n_{contract}} C_m^{contract} C_n^{contract} \left(\frac{\pi}{a_m + a_n}\right)^{3/2} \frac{(2i_x)! (2i_y)! (2i_z)!}{4^L (a_m + a_n)^L i_x! i_y! i_z!} = 1$$
    Assuming the primitive Gaussian function is the pure $x$ component of the primitive shell ($i_x = L, i_y = i_z = 0$), the normalization condition is
    $$\iiint_\infty d\vec{r} (\phi(\vec{r}))^2 = \sum_{m, n}^{n_{contract}} C_m^{contract} C_n^{contract} \left(\frac{\pi}{a_m + a_n}\right)^{3/2} \frac{(2L)!}{4^L (a_m + a_n)^L L!} = 1$$
    If the contraction coefficients $C_m^{contract}$ does not satisfy the condition above, an prefactor of
    $$C^{normalize} = \left( \sum_{m, n}^{n_{contract}} C_m^{contract} C_n^{contract} \left(\frac{\pi}{a_m + a_n}\right)^{3/2} \frac{(2L)!}{4^L (a_m + a_n)^L L!} \right)^{-1/2}$$
    will normalize the contracted Gaussian function.

    Notice that we normalize each primitive Gaussian function first, then the contracted Gaussian.
    """

    assert type(shell.primitive_exponents) is np.ndarray
    assert type(shell.primitive_coefficients) is np.ndarray
    assert shell.primitive_exponents.ndim == 1
    assert shell.primitive_coefficients.ndim == 1
    assert shell.primitive_exponents.shape[0] > 0
    assert shell.primitive_exponents.shape[0] == shell.primitive_coefficients.shape[0]
    L = shell.angular
    assert type(L) is int
    assert L >= 0

    shell.primitive_coefficients = shell.primitive_coefficients * np.power(shell.primitive_exponents, L * 0.5 + 0.75)

    product_prefactor = np.power(np.pi, 1.5) * factorial(2 * L) / (4**L * factorial(L))
    coefficient_product = np.outer(shell.primitive_coefficients, shell.primitive_coefficients)
    exponent_sum = shell.primitive_exponents[:, None] + shell.primitive_exponents[None, :]
    exponent_sum = np.power(exponent_sum, -L - 1.5)
    normalization_constant = product_prefactor * np.sum(coefficient_product * exponent_sum)
    normalization_constant = np.power(normalization_constant, -0.5)

    shell.primitive_coefficients *= normalization_constant
