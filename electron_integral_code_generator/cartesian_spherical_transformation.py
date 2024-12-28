
from math import factorial, floor, cos, sin, sqrt, pi
import sympy

from electron_integral_code_generator.math_constant.binomial_coefficient import binomial_coefficient

numerical_zero_threshold = 1e-14

# https://en.wikipedia.org/wiki/Solid_harmonics#Real_form

def gamma_lkm(l: int, k: int, m: int) -> float:
    return (-1)**k * 2**(-l) * binomial_coefficient(l, k) * binomial_coefficient(2 * l - 2 * k, l) * factorial(l - 2 * k) / factorial(l - 2 * k - m)

def Pi_lm(l: int, m: int) -> sympy.Expr:
    x, y, z, r = sympy.symbols('x y z r')
    expression = 0
    for k in range(int(floor((l - m) / 2)) + 1):
        # expression += gamma_lkm(l, k, m) * r**(2 * k) * z**(l - 2 * k - m)
        expression += gamma_lkm(l, k, m) * (x**2 + y**2 + z**2)**k * z**(l - 2 * k - m)
    return expression

def A_m(m: int) -> sympy.Expr:
    x, y = sympy.symbols('x y')
    expression = 0
    for p in range(m + 1):
        cos_value = cos((m - p) * pi / 2)
        if abs(cos_value) > numerical_zero_threshold:
            expression += binomial_coefficient(m, p) * x**p * y**(m - p) * cos_value
    return expression

def B_m(m: int) -> sympy.Expr:
    x, y = sympy.symbols('x y')
    expression = 0
    for p in range(m + 1):
        sin_value = sin((m - p) * pi / 2)
        if abs(sin_value) > numerical_zero_threshold:
            expression += binomial_coefficient(m, p) * x**p * y**(m - p) * sin_value
    return expression

def C_lm(l: int, m: int) -> sympy.Expr:
    coefficient = sqrt((2 - (1 if m == 0 else 0)) * factorial(l - m) / factorial(l + m))
    return coefficient * Pi_lm(l, m) * A_m(m)

def S_lm(l: int, m: int) -> sympy.Expr:
    coefficient = sqrt(2 * factorial(l - m) / factorial(l + m))
    return coefficient * Pi_lm(l, m) * B_m(m)

def double_factorial_recursive(n: int) -> int:
    if n <= 1:
        return 1
    else:
        return n * double_factorial_recursive(n - 2)

def anti_normalize(expression: sympy.Expr, l: int) -> sympy.Expr:
    x, y, z = sympy.symbols('x y z')
    expression = sympy.expand(expression)
    for i_x in range(l + 1):
        for i_y in range(l - i_x + 1):
            i_z = l - i_x - i_y
            additional_normalization = sqrt(factorial(i_x) * factorial(i_y) * factorial(i_z) / factorial(2 * i_x) / factorial(2 * i_y) / factorial(2 * i_z)
                                            * factorial(2 * l) / factorial(l))
            to_substitute = x**i_x * y**i_y * z**i_z
            expression = expression.subs(to_substitute, to_substitute / additional_normalization)
    return expression

if __name__ == "__main__":
    l = 4
    for m in range(l, 0, -1):
        print(anti_normalize(S_lm(l, m), l))
    for m in range(0, l + 1):
        print(anti_normalize(C_lm(l, m), l))