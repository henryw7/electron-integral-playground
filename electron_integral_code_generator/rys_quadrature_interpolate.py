from scipy import integrate
from numpy.polynomial import Polynomial
from numpy.polynomial.chebyshev import Chebyshev
import numpy as np

integral_error_threshold = 5e-14

def reference_rys_quadrature_roots(x: float, m: int) -> tuple[np.ndarray, np.ndarray]:
    roots = np.zeros((m, m))
    weights = np.zeros((m, m))

    def weight_function(t):
        return np.exp(-x * t**2)
    p_0 = Polynomial(coef = [1])
    t = Polynomial(coef = [0,1])
    t2 = Polynomial(coef = [0,0,1])

    numerical_integral_quadrature_order = m * 3 # This is an approximation!
    pn_pn, _ = integrate.fixed_quad(lambda t: weight_function(t) * p_0(t**2) * p_0(t**2), 0, 1, n = numerical_integral_quadrature_order) # = reference_boys(x, 0)
    pn_t2_pn, _ = integrate.fixed_quad(lambda t: weight_function(t) * p_0(t**2) * p_0(t**2) * t**2, 0, 1, n = numerical_integral_quadrature_order) # = reference_boys(x, 1)
    a_n = pn_t2_pn / pn_pn

    p_1 = (t - a_n) * p_0
    p_n = p_1
    p_n_1 = p_0
    roots_n = p_n(t2).roots()
    roots_n = roots_n[roots_n > 0]
    roots[0,0] = roots_n[0]
    weights[0,0] = (pn_pn / p_n.deriv()(roots_n))[0]

    for i in range(2, m + 1):
        pn_1_pn_1 = pn_pn
        pn_pn, _ = integrate.fixed_quad(lambda t: weight_function(t) * p_n(t**2) * p_n(t**2), 0, 1, n = numerical_integral_quadrature_order)
        pn_t2_pn, _ = integrate.fixed_quad(lambda t: weight_function(t) * p_n(t**2) * p_n(t**2) * t**2, 0, 1, n = numerical_integral_quadrature_order)
        a_n = pn_t2_pn / pn_pn
        b_n = pn_pn / pn_1_pn_1
        p_n_save = p_n
        p_n = (t - a_n) * p_n - b_n * p_n_1
        p_n_1 = p_n_save

        roots_n = p_n(t2).roots()
        roots_n = roots_n[roots_n > 0]
        roots[i - 1, :i] = roots_n
        weights[i - 1, :i] = pn_pn / (p_n_1(roots_n**2) * p_n.deriv()(roots_n**2))

    return roots, weights

def compute_and_save_reference(Lmax):
    Nroot_max = Lmax // 2 + 1

    x = np.arange(1, 2, 0.0001)
    # x = np.arange(0, 60, 0.0001)
    n_point = x.shape[0]
    print(f"n_point = {n_point}")
    y = np.zeros((2, Nroot_max, Nroot_max, n_point))
    for i_point in range(n_point):
        roots, weights = reference_rys_quadrature_roots(x[i_point], Nroot_max)
        y[0, :, :, i_point] = roots
        y[1, :, :, i_point] = weights
        if (i_point % (n_point // 100) == 0):
            print(f"{i_point // (n_point // 100):2d}%")

    np.save("reference_rys_x.npy", x)
    np.save("reference_rys_y.npy", y)

def chebyshev_interpolate(x_start: float, x_end: float, chebyshev_order: int, rys_order: int) -> np.ndarray:
    coefficient = np.zeros((2, chebyshev_order + 1, rys_order, rys_order))
    for root_or_weight in range(2):
        for i in range(rys_order):
            for j in range(i + 1):
                def chebyshev_input(x):
                    x = (x + 1) / 2 * (x_end - x_start) + x_start
                    x = np.atleast_1d(np.asarray(x))
                    n_elements = len(x)
                    y = np.empty(n_elements)
                    for i_element in range(n_elements):
                        roots, weights = reference_rys_quadrature_roots(x[i_element], rys_order)
                        if root_or_weight == 0:
                            y[i_element] = roots[i, j]
                        else:
                            y[i_element] = weights[i, j]
                    return y

                chebyshev_object = Chebyshev.interpolate(chebyshev_input, chebyshev_order)
                polynomial_object = chebyshev_object.convert(kind = Polynomial)
                coefficient_ij_roots = polynomial_object.coef
                if coefficient_ij_roots.shape != (chebyshev_order + 1,):
                    pad_dimension = chebyshev_order + 1 - coefficient_ij_roots.shape[0]
                    coefficient_ij_roots = np.pad(coefficient_ij_roots, (0, pad_dimension), 'constant', constant_values = (0,0))
                coefficient[root_or_weight, :, i, j] = coefficient_ij_roots
    return coefficient

def boys_chebyshev_evaluate(x: np.ndarray, polynomial_coefficients: np.ndarray, x_start: float, x_end: float) -> np.ndarray:
    assert polynomial_coefficients.ndim == 4
    assert polynomial_coefficients.shape[0] == 2
    chebyshev_order = polynomial_coefficients.shape[1] - 1
    rys_order = polynomial_coefficients.shape[2]
    assert rys_order == polynomial_coefficients.shape[3]

    x = (x - x_start) / (x_end - x_start) * 2 - 1
    x = np.atleast_1d(np.asarray(x))
    n_elements = len(x)
    y = np.zeros((2, rys_order, rys_order, n_elements))
    for root_or_weight in range(2):
        for i in range(rys_order):
            for j in range(i + 1):
                for i_chebyshev in range(chebyshev_order + 1):
                    y[root_or_weight, i, j, :] += polynomial_coefficients[root_or_weight, i_chebyshev, i, j] * x**i_chebyshev
    return y

if __name__ == "__main__":
    Lmax = 26
    Nroot_max = Lmax // 2 + 1
    # compute_and_save_reference(Lmax)

    x_range = [1,1.1]
    coefficient = chebyshev_interpolate(x_range[0], x_range[1], 5, Nroot_max)
    reference_rys_x = np.load("reference_rys_x.npy")
    reference_rys_y = np.load("reference_rys_y.npy")
    x_where = (reference_rys_x >= x_range[0]) & (reference_rys_x <= x_range[1])

    chebyshev_y = boys_chebyshev_evaluate(reference_rys_x[x_where], coefficient, x_range[0], x_range[1])
    reference_y = reference_rys_y[:, :, :, x_where]
    print(np.max(np.abs((chebyshev_y[1, 13, 13, :] - reference_y[1, 13, 13, :]) / reference_y[1, 13, 13, :])))

    # np.set_printoptions(linewidth = np.iinfo(np.int32).max, threshold = np.iinfo(np.int32).max, precision = 16, suppress = True)
    # print(f"test = np.{repr(chebyshev_y[1, 13, 13, :])}")
    # print(f"ref  = np.{repr(reference_y[1, 13, 13, :])}")
