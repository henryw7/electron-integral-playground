from scipy import integrate
from numpy.polynomial import Polynomial
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

if __name__ == "__main__":
    np.set_printoptions(linewidth = np.iinfo(np.int32).max, threshold = np.iinfo(np.int32).max, precision = 16, suppress = True)
    roots, weights = reference_rys_quadrature_roots(4.0, 8)
    print(roots)
    print(weights)
