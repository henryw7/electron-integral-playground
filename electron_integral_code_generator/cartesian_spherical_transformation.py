
from math import factorial, floor, cos, sin, sqrt, pi
import sympy

from electron_integral_code_generator.math_constant.binomial_coefficient import binomial_coefficient

numerical_zero_threshold = 1e-14

# $$\gamma_{l,k,m} = (-1)^k 2^{-l} \left({\begin{array}{*{20}c} l \\ k \end{array}}\right) \left({\begin{array}{*{20}c} 2l - 2k \\ l \end{array}}\right) \frac{(l - 2k)!}{(l - 2k - m)!}$$
def gamma_lkm(l: int, k: int, m: int) -> float:
    return (-1)**k * 2**(-l) * binomial_coefficient(l, k) * binomial_coefficient(2 * l - 2 * k, l) * factorial(l - 2 * k) / factorial(l - 2 * k - m)

# $$\Pi_{l,m}(z, r) = \sum_{k = 0}^{floor((l - m) / 2)} \gamma_{l,k,m} r^{2k} z^{l -2k - m}$$
def Pi_lm(l: int, m: int) -> sympy.Expr:
    x, y, z, r = sympy.symbols('x y z r')
    expression = 0
    for k in range(int(floor((l - m) / 2)) + 1):
        # expression += gamma_lkm(l, k, m) * r**(2 * k) * z**(l - 2 * k - m)
        expression += gamma_lkm(l, k, m) * (x**2 + y**2 + z**2)**k * z**(l - 2 * k - m)
    return expression

# $$A_m(x, y) = \sum_{p = 0}^m \left({\begin{array}{*{20}c} m \\ p \end{array}}\right) x^p y^{m - p} cos\left(\frac{\pi}{2} (m - p)\right)$$
def A_m(m: int) -> sympy.Expr:
    x, y = sympy.symbols('x y')
    expression = 0
    for p in range(m + 1):
        cos_value = cos((m - p) * pi / 2)
        if abs(cos_value) > numerical_zero_threshold:
            expression += binomial_coefficient(m, p) * x**p * y**(m - p) * cos_value
    return expression

# $$B_m(x, y) = \sum_{p = 0}^m \left({\begin{array}{*{20}c} m \\ p \end{array}}\right) x^p y^{m - p} sin\left(\frac{\pi}{2} (m - p)\right)$$
def B_m(m: int) -> sympy.Expr:
    x, y = sympy.symbols('x y')
    expression = 0
    for p in range(m + 1):
        sin_value = sin((m - p) * pi / 2)
        if abs(sin_value) > numerical_zero_threshold:
            expression += binomial_coefficient(m, p) * x**p * y**(m - p) * sin_value
    return expression

# $$C_{l, m}(x, y, z) = \sqrt{\frac{(2 - \delta_{m, 0}) (l - m)!}{(l + m)!}} \Pi_{l,m}(z, r) A_m(x, y) \qquad m = 0,1,...,l$$
def C_lm(l: int, m: int) -> sympy.Expr:
    coefficient = sqrt((2 - (1 if m == 0 else 0)) * factorial(l - m) / factorial(l + m))
    return coefficient * Pi_lm(l, m) * A_m(m)

# $$S_{l, m}(x, y, z) = \sqrt{\frac{2(l - m)!}{(l + m)!}} \Pi_{l,m}(z, r) B_m(x, y) \qquad m = 1,2,...,l$$
def S_lm(l: int, m: int) -> sympy.Expr:
    coefficient = sqrt(2 * factorial(l - m) / factorial(l + m))
    return coefficient * Pi_lm(l, m) * B_m(m)

# This gaurantees that the diagonal elements of the overlap matrix are all one in both Cartesian and spherical orbitals
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

def get_cartesian_to_spherical_equations(l: int) -> list[sympy.Expr]:
    equations = []
    for m in range(l, 0, -1):
        equations.append(anti_normalize(S_lm(l, m), l))
    for m in range(0, l + 1):
        equations.append(anti_normalize(C_lm(l, m), l))
    return equations

if __name__ == "__main__":
    l = 6
    equations = get_cartesian_to_spherical_equations(l)
    for eq in equations:
        print(eq)