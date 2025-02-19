
import numpy as np
from math import gamma
from scipy.special import gammainc
from scipy.interpolate import CubicSpline
from numpy.polynomial import Polynomial
from numpy.polynomial.chebyshev import Chebyshev

np.seterr(invalid='ignore') # Suppress divide by zero error

def reference_boys(x: float, m: int) -> float:
    return gamma(0.5 + m) * gammainc(0.5 + m, x) / (2 * x**(0.5 + m))

def boys_cubic_spline_interpolate(x_start: float, x_end: float, x_interval: float, order: int) -> np.ndarray:
    x = np.arange(0, x_end + 2.0 + x_interval * 3, x_interval)
    y = reference_boys(x, order)
    y[0] = 1 / (2.0 * order + 1)
    dy = -1 / (2.0 * order + 3)
    spline = CubicSpline(x, y, bc_type=((1, dy), (1, 0.0)))
    required_range = np.where((spline.x >= x_start) & (spline.x <= x_end))[0]
    spline_c = spline.c[::-1, required_range]
    return spline_c

def boys_cubic_spline_evaluate(x: np.ndarray, spline_c: np.ndarray, x_start: float, x_interval: float) -> np.ndarray:
    i = np.floor((x - x_start) / x_interval)
    i = i.astype(np.int32)
    c = spline_c[:, i]
    x0 = x_start + x_interval * i
    return c[0] + c[1] * (x - x0) + c[2] * (x - x0)**2 + c[3] * (x - x0)**3

def get_cublic_spline_coefficients(Lmax):
    x_start = 0.2
    x_end = 15.0
    x_interval = 0.005
    x_interval_tight = 1e-5

    print(f"coefficient size = {int(np.round((x_end - x_start) / x_interval + 1))} * {Lmax + 1}")

    coefficient_file = open("spline_coefficent.txt", "w")

    for order in range(0, Lmax + 1, 1):
        interpolate_coefficient = boys_cubic_spline_interpolate(x_start, x_end, x_interval, order)
        
        x_dense = np.arange(x_start, x_end + x_interval_tight, x_interval_tight)
        y_approximate = boys_cubic_spline_evaluate(x_dense, interpolate_coefficient, x_start, x_interval)
        y_exact = reference_boys(x_dense, order)
        max_relative_error = np.max(np.abs((y_approximate - y_exact) / y_exact))

        print(max_relative_error)
        coefficient_text = "{ "
        for i in range(interpolate_coefficient.shape[1]):
            for j in range(interpolate_coefficient.shape[0]):
                coefficient_text += f"{interpolate_coefficient[j, i]:.16e}, "
            if (i % 2 == 1):
                coefficient_text += "\n  "
        coefficient_text += "},\n"

        coefficient_file.write(coefficient_text)

    coefficient_file.close()

def boys_chebyshev_interpolate(x_start: float, x_end: float, chebyshev_order: int, boys_order: int) -> np.ndarray:
    def chebyshev_input(x):
        x = (x + 1) / 2 * (x_end - x_start) + x_start
        return reference_boys(x, boys_order)
    chebyshev_object = Chebyshev.interpolate(chebyshev_input, chebyshev_order)
    polynomial_object = chebyshev_object.convert(kind = Polynomial)
    return polynomial_object.coef

def boys_chebyshev_evaluate(x: np.ndarray, polynomial_coefficients: np.ndarray, x_start: float, x_end: float) -> np.ndarray:
    chebyshev_order = polynomial_coefficients.shape[0] - 1
    x = (x - x_start) / (x_end - x_start) * 2 - 1
    y = np.zeros_like(x)
    for i in range(chebyshev_order + 1):
        y += polynomial_coefficients[i] * x**i
    return y

def get_chebyshev_coefficients(Lmax):
    x_start = 0
    x_end = 20.0
    x_interval = 1.0
    x_interval_tight = 1e-5
    n_interval = int(np.round((x_end - x_start) / x_interval))
    chebyshev_order = 10

    print(f"coefficient size = {n_interval} * {chebyshev_order + 1} * {Lmax + 1}")

    # coefficient_file = open("chebyshev_coefficent.txt", "w")

    for boys_order in range(0, Lmax + 1, 1):
        polynomial_coefficients = np.empty((n_interval, chebyshev_order + 1))
        for i_interval in range(n_interval):
            polynomial_coefficients[i_interval, :] = boys_chebyshev_interpolate(x_start + i_interval * x_interval, x_start + (i_interval + 1) * x_interval, chebyshev_order, boys_order)

        x_dense = np.arange(x_start, x_end + x_interval_tight, x_interval_tight)
        y_exact = reference_boys(x_dense, boys_order)
        y_exact[0] = 1 / (2.0 * boys_order + 1)

        y_approximate = np.zeros_like(x_dense)
        for i_interval in range(n_interval):
            i_in_interval = np.where(np.logical_and(x_start + i_interval * x_interval <= x_dense, x_dense <= x_start + (i_interval + 1) * x_interval))
            y_approximate[i_in_interval] = boys_chebyshev_evaluate(x_dense[i_in_interval], polynomial_coefficients[i_interval, :], x_start + i_interval * x_interval, x_start + (i_interval + 1) * x_interval)
        max_relative_error = np.max(np.abs((y_approximate - y_exact) / y_exact))

        print(max_relative_error)
    #     coefficient_text = "{ "
    #     for i in range(interpolate_coefficient.shape[1]):
    #         for j in range(interpolate_coefficient.shape[0]):
    #             coefficient_text += f"{interpolate_coefficient[j, i]:.16e}, "
    #         if (i % 2 == 1):
    #             coefficient_text += "\n  "
    #     coefficient_text += "},\n"

    #     coefficient_file.write(coefficient_text)

    # coefficient_file.close()

if __name__ == "__main__":
    Lmax = 26
    # get_cublic_spline_coefficients(Lmax)
    get_chebyshev_coefficients(Lmax)
