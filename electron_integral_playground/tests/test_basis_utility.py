
import pytest

import numpy as np

from electron_integral_playground.basis import GaussianShell, normalize_shell

def test_basis_normalization_multiple_primitives():
    shell = GaussianShell(
        angular = 0,
        spherical = True,
        i_ao_start = -1,
        i_atom = -1,
        primitive_exponents = np.array([0.3683820000E+02, 0.5481720000E+01, 0.1113270000E+01]),
        primitive_coefficients = np.array([0.6966866381E-01, 0.3813463493E+00, 0.6817026244E+00]),
    )
    normalize_shell(shell)
    print("S shell")
    print(shell)

    normalize_shell(shell)
    print("S shell again")
    print(shell)

    shell.angular = 1
    normalize_shell(shell)
    print("P shell")
    print(shell)

    shell.angular = 2
    normalize_shell(shell)
    print("D shell")
    print(shell)

    shell.angular = 3
    normalize_shell(shell)
    print("F shell")
    print(shell)

    shell.angular = 3
    normalize_shell(shell)
    print("F shell again")
    print(shell)

    raise NotImplementedError("Reference answer not available yet")

def test_basis_normalization_one_primitive():
    shell = GaussianShell(
        angular = 0,
        spherical = True,
        i_ao_start = -1,
        i_atom = -1,
        primitive_exponents = np.array([0.1243280000E+00]),
        primitive_coefficients = np.array([1.0]),
    )
    normalize_shell(shell)
    print("S shell")
    print(shell)

    normalize_shell(shell)
    print("S shell again")
    print(shell)

    shell.angular = 1
    normalize_shell(shell)
    print("P shell")
    print(shell)

    shell.angular = 2
    normalize_shell(shell)
    print("D shell")
    print(shell)

    shell.angular = 3
    normalize_shell(shell)
    print("F shell")
    print(shell)

    shell.angular = 3
    normalize_shell(shell)
    print("F shell again")
    print(shell)

    raise NotImplementedError("Reference answer not available yet")
