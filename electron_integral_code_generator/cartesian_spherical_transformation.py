
from math import factorial, floor, cos, sin, sqrt, pi
import sympy

from electron_integral_code_generator.math_constant.binomial_coefficient import binomial_coefficient
from electron_integral_code_generator.sympy_utility import sympy_number_to_cpp_float

# $$\gamma_{l,k,m} = (-1)^k 2^{-l} \left({\begin{array}{*{20}c} l \\ k \end{array}}\right) \left({\begin{array}{*{20}c} 2l - 2k \\ l \end{array}}\right) \frac{(l - 2k)!}{(l - 2k - m)!}$$
def gamma_lkm(l: int, k: int, m: int) -> sympy.Expr:
    numerator = (-1)**k * binomial_coefficient(l, k) * binomial_coefficient(2 * l - 2 * k, l) * factorial(l - 2 * k)
    denominator = factorial(l - 2 * k - m) * 2**l
    return sympy.simplify(f"{numerator} / {denominator}")

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
        expression += binomial_coefficient(m, p) * x**p * y**(m - p) * sympy.simplify(f"cos(({m - p}) * pi / 2)")
    return expression

# $$B_m(x, y) = \sum_{p = 0}^m \left({\begin{array}{*{20}c} m \\ p \end{array}}\right) x^p y^{m - p} sin\left(\frac{\pi}{2} (m - p)\right)$$
def B_m(m: int) -> sympy.Expr:
    x, y = sympy.symbols('x y')
    expression = 0
    for p in range(m + 1):
        expression += binomial_coefficient(m, p) * x**p * y**(m - p) * sympy.simplify(f"sin(({m - p}) * pi / 2)")
    return expression

# $$C_{l, m}(x, y, z) = \sqrt{\frac{(2 - \delta_{m, 0}) (l - m)!}{(l + m)!}} \Pi_{l,m}(z, r) A_m(x, y) \qquad m = 0,1,...,l$$
def C_lm(l: int, m: int) -> sympy.Expr:
    numerator = (2 - (1 if m == 0 else 0)) * factorial(l - m)
    denominator = factorial(l + m)
    coefficient = sympy.simplify(f"sqrt({numerator} / {denominator})")
    return coefficient * Pi_lm(l, m) * A_m(m)

# $$S_{l, m}(x, y, z) = \sqrt{\frac{2(l - m)!}{(l + m)!}} \Pi_{l,m}(z, r) B_m(x, y) \qquad m = 1,2,...,l$$
def S_lm(l: int, m: int) -> sympy.Expr:
    numerator = 2 * factorial(l - m)
    denominator = factorial(l + m)
    coefficient = sympy.simplify(f"sqrt({numerator} / {denominator})")
    return coefficient * Pi_lm(l, m) * B_m(m)

# This gaurantees that the diagonal elements of the overlap matrix are all one in both Cartesian and spherical orbitals
def anti_normalize(expression: sympy.Expr, l: int) -> sympy.Expr:
    x, y, z = sympy.symbols('x y z')
    placeholder = sympy.symbols('placeholder') # Make sure each subsitution happens only once
    expression = sympy.expand(expression)
    for i_x in range(l + 1):
        for i_y in range(l - i_x + 1):
            i_z = l - i_x - i_y
            numerator = factorial(i_x) * factorial(i_y) * factorial(i_z) * factorial(2 * l)
            denominator = factorial(2 * i_x) * factorial(2 * i_y) * factorial(2 * i_z) * factorial(l)
            additional_normalization = sympy.simplify(f"sqrt({numerator} / {denominator})")
            to_substitute = x**i_x * y**i_y * z**i_z
            expression = expression.subs(to_substitute, placeholder / additional_normalization)
            expression = expression.subs(placeholder, to_substitute)
    return expression

def transformation_expression_to_vector(expression: sympy.Expr, l: int) -> list[sympy.Expr]:
    x, y, z = sympy.symbols('x y z')
    expression = sympy.expand(expression)
    transformation_vector = []
    for i_x in range(l, -1, -1):
        for i_y in range(l - i_x, -1, -1):
            i_z = l - i_x - i_y
            term = x**i_x * y**i_y * z**i_z
            prefactor = expression.subs(term, 1)
            prefactor = prefactor.subs(x, 0)
            prefactor = prefactor.subs(y, 0)
            prefactor = prefactor.subs(z, 0)
            transformation_vector.append(prefactor)
    return transformation_vector

def get_cartesian_to_spherical_equations(l: int) -> list[sympy.Expr]:
    equations = []
    for m in range(l, 0, -1):
        equations.append(anti_normalize(S_lm(l, m), l))
    for m in range(0, l + 1):
        equations.append(anti_normalize(C_lm(l, m), l))
    return equations

if __name__ == "__main__":
    l = 9
    equations = get_cartesian_to_spherical_equations(l)

    transformation_matrix = []
    for eq in equations:
        transformation_vector = transformation_expression_to_vector(eq, l)
        for i_term, term in enumerate(transformation_vector):
            transformation_vector[i_term] = sympy_number_to_cpp_float(term)
        transformation_matrix.append(transformation_vector)

    max_term_length = 0
    for i in range(len(transformation_matrix)):
        for j in range(len(transformation_matrix[i])):
            term_length = len(transformation_matrix[i][j])
            max_term_length = max(max_term_length, term_length)

    for i in range(len(transformation_matrix)):
        for j in range(len(transformation_matrix[i])):
            term = transformation_matrix[i][j]
            print(f"{term:>{max_term_length + 1}s},", end = "")
        print()

    print()

    for i in range(len(transformation_matrix)):
        line = ""
        for j in range(len(transformation_matrix[i])):
            term = transformation_matrix[i][j]
            if term == "0":
                continue
            elif term == "1.0":
                line += f" + vector[{j} * increment]"
            elif term.startswith("-"):
                line += f" - {term[1:]} * vector[{j} * increment]"
            else:
                line += f" + {term} * vector[{j} * increment]"

        if line.startswith(" - "):
            line = "-" + line[3:]
        else:
            assert line.startswith(" + ")
            line = line[3:]
        print(line)
