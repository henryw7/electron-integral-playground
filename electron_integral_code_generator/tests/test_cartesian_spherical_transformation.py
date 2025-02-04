
import pytest

from sympy import expand, simplify
from electron_integral_code_generator.cartesian_spherical_transformation import get_cartesian_to_spherical_equations

def test_cartesian_to_spherical_d_orbital():
    test_equations = get_cartesian_to_spherical_equations(2)
    ref_equations = [
        expand("x*y"),
        expand("y*z"),
        expand("z^2 - 1/2 * (x^2 + y^2)"),
        expand("x*z"),
        expand("sqrt(3)/2 * (x^2 - y^2)"),
    ]

    assert len(test_equations) == len(ref_equations)
    for i in range(len(ref_equations)):
        assert simplify(test_equations[i] - ref_equations[i]) == 0

def test_cartesian_to_spherical_f_orbital():
    test_equations = get_cartesian_to_spherical_equations(3)
    ref_equations = [
        expand("3/2/sqrt(2) * x^2*y - 1/2*sqrt(5/2) * y^3"),
        expand("x*y*z"),
        expand("-1/2*sqrt(3/10) * x^2*y - 1/2*sqrt(3/2) * y^3 + 2*sqrt(3/10) * y*z^2"),
        expand("-3/2/sqrt(5) * (x^2*z + y^2*z) + z^3"),
        expand("-1/2*sqrt(3/2) * x^3 - 1/2*sqrt(3/10) * x*y^2 + 2*sqrt(3/10) * x*z^2"),
        expand("sqrt(3)/2 * (x^2*z - y^2*z)"),
        expand("1/2*sqrt(5/2) * x^3 - 3/2/sqrt(2) * x*y^2"),
    ]

    assert len(test_equations) == len(ref_equations)
    for i in range(len(ref_equations)):
        assert simplify(test_equations[i] - ref_equations[i]) == 0
