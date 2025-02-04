
from math import factorial
from fractions import Fraction

if __name__ == "__main__":
    for L in range(9 + 1):
        inverse_L_normalization = Fraction(factorial(2 * L), factorial(L))
        for i_x in range(L, -1, -1):
            for i_y in range(L - i_x, -1, -1):
                i_z = L - i_x - i_y
                xyz_normalization = Fraction(factorial(i_x) * factorial(i_y) * factorial(i_z),
                                            factorial(2 * i_x) * factorial(2 * i_y) * factorial(2 * i_z))
                kernel_normalization = inverse_L_normalization * xyz_normalization

                if kernel_normalization == 1:
                    kernel_normalization = "1.0"
                elif kernel_normalization.is_integer():
                    kernel_normalization = f"sqrt({kernel_normalization}.0)"
                else:
                    kernel_normalization = f"sqrt({kernel_normalization.numerator}.0/{kernel_normalization.denominator}.0)"
                print(f"{kernel_normalization:>18s}, ", end = "")
        print()
