
from electron_integral_code_generator.math_constant.double_factorial import double_factorial

from math import factorial
from fractions import Fraction

def boys_infinity_asymptotic_prefactor(m: int) -> Fraction:
    if m == 0: return Fraction(1,2)
    return Fraction(double_factorial(2 * m - 1), 2**(m + 1))

def boys_zero_asymptotic_prefactor(m: int, k: int) -> Fraction:
    return (-1)**k * Fraction(1, factorial(k) * (2 * m + 2 * k + 1))

if __name__ == "__main__":
    L_max = 24 + 2
    print("Asymptotic to infinity:")
    for m in range(L_max + 1):
        prefactor = boys_infinity_asymptotic_prefactor(m)
        print(f"{prefactor.numerator}.0/{prefactor.denominator}.0*sqrt(M_PI),")

    taylor_order = 8
    print("Asymptotic to zero:")
    for m in range(L_max + 1):
        print("{ ", end = "")
        for k in range(taylor_order + 1):
            prefactor = boys_zero_asymptotic_prefactor(m, k)
            prefactor = f"{prefactor}.0" if prefactor.is_integer() else f"{prefactor.numerator}.0/{prefactor.denominator}.0"
            print(f"{prefactor:>13s}, ", end = "")
        print("},")

