
from math import factorial, floor

def hermite_polynomial_coefficient(n: int, m: int) -> int:
    return factorial(n) // (factorial(m) * factorial(n - 2 * m) * 2**m)

def hermite_polynomial_coefficients(n: int) -> list[int]:
    coefficients = []
    for m in range(n // 2 + 1):
        coefficients.append(hermite_polynomial_coefficient(n, m))
    return coefficients

if __name__ == "__main__":
    for n in range(26 + 1):
        coefficients = hermite_polynomial_coefficients(n)
        coefficients.extend([0] * (n + 1 - len(coefficients)))
        coefficients.reverse()
        for coefficient in coefficients:
            print(f"{coefficient:15d}, ", end = "")
        print()
