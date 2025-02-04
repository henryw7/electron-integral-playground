
import pytest

import sympy

from electron_integral_code_generator.sympy_utility import simplify, remove_whitespace, sympy_number_to_cpp_float

def test_power_operation():
    term = simplify("(a + b)**3")
    assert remove_whitespace(term) == remove_whitespace("(a+b) * (a+b) * (a+b)")

    term = simplify("x**2 + x_and_some_random_long_name_with_numbers_like_1234**2")
    assert remove_whitespace(term) == remove_whitespace("x * x + x_and_some_random_long_name_with_numbers_like_1234 * x_and_some_random_long_name_with_numbers_like_1234")

    term = simplify("(1 + x**2)**3")
    assert remove_whitespace(term) == remove_whitespace("(1 + x*x) * (1 + x*x) * (1 + x*x)") or \
           remove_whitespace(term) == remove_whitespace("(x*x + 1) * (x*x + 1) * (x*x + 1)")

    term = simplify("(((x - 1)**3 + y**2)**2 + 3)**2")
    assert remove_whitespace(term) == remove_whitespace("((y*y+(x-1)*(x-1)*(x-1))*(y*y+(x-1)*(x-1)*(x-1))+3)*((y*y+(x-1)*(x-1)*(x-1))*(y*y+(x-1)*(x-1)*(x-1))+3)")

    term = simplify("x**12 + 1")
    assert remove_whitespace(term) == remove_whitespace("x*x*x*x*x*x*x*x*x*x*x*x + 1")

    term = simplify("-x**2 + 1")
    assert remove_whitespace(term) == remove_whitespace("-x*x + 1") or \
           remove_whitespace(term) == remove_whitespace("1 - x*x")

def test_constant_conversion():
    term = sympy.simplify("-1")
    term = sympy_number_to_cpp_float(term)
    assert term == "-1.0"

    term = sympy.simplify("499/10")
    term = sympy_number_to_cpp_float(term)
    assert term == "499.0/10.0"

    term = sympy.simplify("sqrt(3/10)")
    term = sympy_number_to_cpp_float(term)
    assert term == "sqrt(30.0)/10.0"

    term = sympy.simplify("sqrt(70)*7")
    term = sympy_number_to_cpp_float(term)
    assert term == "7.0*sqrt(70.0)"
