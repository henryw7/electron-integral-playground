
from electron_integral_code_generator.sympy_utility import simplify

def recur_mcmurchie_davidson_E_term(i: int, j: int, t: int, xyz: str) -> str:
    r"""
    Apply the McMurchie-Davidson recurrence relationship:
    \begin{align*}
        E_t^{i+1,j} &= \frac{1}{2p} E_{t-1}^{ij} + (P_x - A_x)E_t^{ij} + (t+1)E_{t+1}^{ij} \\
        E_t^{i,j+1} &= \frac{1}{2p} E_{t-1}^{ij} + (P_x - B_x)E_t^{ij} + (t+1)E_{t+1}^{ij}
    \end{align*}
    with the following base cases:
    \begin{align*}
        E_0^{00} &= 1 \\
        E_t^{ij} &= 0 \quad if \quad t < 0 \ or \ t > i+j
    \end{align*}
    """
    assert xyz in [ "x", "y", "z" ]
    assert i >= 0 and j >= 0
    if t < 0 or t > i + j:
        return None
    if i == 0 and j == 0 and t == 0:
        return "1"
    if j > 0:
        E_t_minus_1 = recur_mcmurchie_davidson_E_term(i, j-1, t-1, xyz)
        E_t_equal   = recur_mcmurchie_davidson_E_term(i, j-1, t,   xyz)
        E_t_plus_1  = recur_mcmurchie_davidson_E_term(i, j-1, t+1, xyz)
        output = ""
        if E_t_minus_1 is not None: output += f" + one_over_two_p * ({E_t_minus_1})"
        if E_t_equal   is not None: output += f" + PB{xyz} * ({E_t_equal})"
        if E_t_plus_1  is not None: output += f" + {t + 1} * ({E_t_plus_1})"
        return output
    else:
        E_t_minus_1 = recur_mcmurchie_davidson_E_term(i-1, j, t-1, xyz)
        E_t_equal   = recur_mcmurchie_davidson_E_term(i-1, j, t,   xyz)
        E_t_plus_1  = recur_mcmurchie_davidson_E_term(i-1, j, t+1, xyz)
        output = ""
        if E_t_minus_1 is not None: output += f" + one_over_two_p * ({E_t_minus_1})"
        if E_t_equal   is not None: output += f" + PA{xyz} * ({E_t_equal})"
        if E_t_plus_1  is not None: output += f" + {t + 1} * ({E_t_plus_1})"
        return output

def get_mcmurchie_davidson_E_term(i: int, j: int, t: int, xyz: str) -> str:
    assert i >= 0 and j >= 0
    assert t >= 0 and t <= i + j
    assert xyz in [ "x", "y", "z" ]
    term = recur_mcmurchie_davidson_E_term(i, j, t, xyz)
    assert term is not None

    return simplify(term)
    
