
from dataclasses import dataclass

import numpy as np
from math import factorial

@dataclass
class GaussianShell:
    angular: int
    spherical: bool
    i_ao_start: int
    i_atom: int
    primitive_exponents: np.ndarray
    primitive_coefficients: np.ndarray

angular_letter_to_l_value_map = { "S":0, "P":1, "D":2, "F":3, "G":4, "H":5, "I":6 }
def angular_letter_to_l_value(letter: str) -> int:
    assert len(letter) == 1
    letter = letter.upper()
    assert letter in angular_letter_to_l_value_map
    return angular_letter_to_l_value_map[letter]

def normalize_shell(shell: GaussianShell) -> None:
    """
    Modify the primitive_coefficients field of shell, to make sure the diagonal of the molecular overlap matrix is 1.
    For a non-S shell, the primitive are normalized according to the Cartesian pure x component (px, dxx, fxxx, gxxxx, etc.).
    This operation is idempotent.
    """
    """
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

    product_prefactor = np.power(np.pi, 1.5) * factorial(2 * L) / (4**L * factorial(L))
    coefficient_product = np.outer(shell.primitive_coefficients, shell.primitive_coefficients)
    exponent_sum = shell.primitive_exponents[:, None] + shell.primitive_exponents[None, :]
    exponent_sum = np.power(exponent_sum, -L - 1.5)
    normalization_constant = product_prefactor * np.sum(coefficient_product * exponent_sum)
    normalization_constant = np.power(normalization_constant, -0.5)

    shell.primitive_coefficients *= normalization_constant
