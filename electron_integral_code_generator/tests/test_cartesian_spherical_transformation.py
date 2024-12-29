
import pytest

from sympy import expand, nsimplify
from electron_integral_code_generator.cartesian_spherical_transformation import get_cartesian_to_spherical_equations

def test_cartesian_to_spherical_d_orbital():
    test_equations = get_cartesian_to_spherical_equations(2)
    ref_equations = [
        expand("x*y"),
        expand("y*z"),
        expand("z^2 - 0.5 * (x^2 + y^2)"),
        expand("x*z"),
        expand("sqrt(3.0)/2.0 * (x^2 - y^2)"),
    ]

    assert len(test_equations) == len(ref_equations)
    for i in range(len(ref_equations)):
        assert nsimplify(test_equations[i] - ref_equations[i], tolerance = 1e-14) == 0

def test_cartesian_to_spherical_f_orbital():
    test_equations = get_cartesian_to_spherical_equations(3)
    ref_equations = [
        expand("1.5/sqrt(2.0) * x^2*y - 0.5*sqrt(2.5) * y^3"),
        expand("x*y*z"),
        expand("-0.5*sqrt(0.3) * x^2*y - 0.5*sqrt(1.5) * y^3 + 2.0*sqrt(0.3) * y*z^2"),
        expand("-1.5/sqrt(5.0) * (x^2*z + y^2*z) + z^3"),
        expand("-0.5*sqrt(1.5) * x^3 - 0.5*sqrt(0.3) * x*y^2 + 2.0*sqrt(0.3) * x*z^2"),
        expand("sqrt(3.0)/2.0 * (x^2*z - y^2*z)"),
        expand("0.5*sqrt(2.5) * x^3 - 1.5/sqrt(2.0) * x*y^2"),
    ]

    assert len(test_equations) == len(ref_equations)
    for i in range(len(ref_equations)):
        assert nsimplify(test_equations[i] - ref_equations[i], tolerance = 1e-14) == 0
