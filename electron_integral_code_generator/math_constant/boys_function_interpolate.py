
import numpy as np
from math import gamma
from scipy.special import gammainc
from scipy.interpolate import CubicSpline


def reference_boys(x: float, m: int) -> float:
    return gamma(0.5 + m) * gammainc(0.5 + m, x) / (2 * x**(0.5 + m))

def boys_interpolate(x_start: float, x_end: float, x_interval: float, order: int) -> np.ndarray:
    x = np.arange(0, x_end + 2.0 + x_interval * 3, x_interval)
    y = reference_boys(x, order)
    y[0] = 1 / (2.0 * order + 1)
    dy = -1 / (2.0 * order + 3)
    spline = CubicSpline(x, y, bc_type=((1, dy), (1, 0.0)))
    required_range = np.where((spline.x >= x_start) & (spline.x <= x_end))[0]
    spline_c = spline.c[::-1, required_range]
    return spline_c

def boys_evaluate_spline(x: np.ndarray, spline_c: np.ndarray, x_start: float, x_interval: float) -> np.ndarray:
    i = np.floor((x - x_start) / x_interval)
    i = i.astype(np.int32)
    c = spline_c[:, i]
    x0 = x_start + x_interval * i
    return c[0] + c[1] * (x - x0) + c[2] * (x - x0)**2 + c[3] * (x - x0)**3

if __name__ == "__main__":
    x_start = 0.2
    x_end = 15.0
    x_interval = 0.005
    x_interval_tight = 1e-5

    print(f"coefficient size = {int(np.round((x_end - x_start) / x_interval + 1))}")

    coefficient_file = open("spline_coefficent.txt", "w")

    for order in range(0, 26 + 1, 1):
        interpolate_coefficient = boys_interpolate(x_start, x_end, x_interval, order)
        
        x_dense = np.arange(x_start, x_end + x_interval_tight, x_interval_tight)
        y_approximate = boys_evaluate_spline(x_dense, interpolate_coefficient, x_start, x_interval)
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