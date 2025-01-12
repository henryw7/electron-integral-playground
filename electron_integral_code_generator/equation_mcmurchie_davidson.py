
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

def recur_mcmurchie_davidson_R_term(tx: int, ty: int, tz: int, n: int) -> str:
    r"""
    Apply the McMurchie-Davidson recurrence relationship:
    \begin{align*}
		R_{t_x+1, t_y, t_z}^n &= t_x R_{t_x-1, t_y, t_z}^{n+1} + PQ_x R_{t_x, t_y, t_z}^{n+1} \\
		R_{t_x, t_y+1, t_z}^n &= t_y R_{t_x, t_y-1, t_z}^{n+1} + PQ_y R_{t_x, t_y, t_z}^{n+1} \\
		R_{t_x, t_y, t_z+1}^n &= t_z R_{t_x, t_y, t_z-1}^{n+1} + PQ_z R_{t_x, t_y, t_z}^{n+1}
    \end{align*}
    with the following base cases:
    \begin{align*}
		R_{000}^n &= (-2\zeta)^n F_n\left(\zeta |\vec{PQ}|^2\right) \\
		R_{t_x, t_y, t_z}^n &= 0 \quad if \quad t_x < 0 \ or \ t_y < 0 \ or \ t_z < 0
    \end{align*}
    """
    assert n >= 0
    if tx < 0 or ty < 0 or tz < 0:
        return None
    if tx == 0 and ty == 0 and tz == 0:
        return f"R_0_0_0_{n}"
    if tz > 0:
        R_t_minus_2 = recur_mcmurchie_davidson_R_term(tx, ty, tz - 2, n + 1)
        R_t_minus_1 = recur_mcmurchie_davidson_R_term(tx, ty, tz - 1, n + 1)
        output = ""
        if R_t_minus_2 is not None: output += f" + {tz - 1} * ({R_t_minus_2})"
        if R_t_minus_1 is not None: output += f" + PQz * ({R_t_minus_1})"
        return output
    elif ty > 0:
        R_t_minus_2 = recur_mcmurchie_davidson_R_term(tx, ty - 2, tz, n + 1)
        R_t_minus_1 = recur_mcmurchie_davidson_R_term(tx, ty - 1, tz, n + 1)
        output = ""
        if R_t_minus_2 is not None: output += f" + {ty - 1} * ({R_t_minus_2})"
        if R_t_minus_1 is not None: output += f" + PQy * ({R_t_minus_1})"
        return output
    else:
        R_t_minus_2 = recur_mcmurchie_davidson_R_term(tx - 2, ty, tz, n + 1)
        R_t_minus_1 = recur_mcmurchie_davidson_R_term(tx - 1, ty, tz, n + 1)
        output = ""
        if R_t_minus_2 is not None: output += f" + {tx - 1} * ({R_t_minus_2})"
        if R_t_minus_1 is not None: output += f" + PQx * ({R_t_minus_1})"
        return output

def get_mcmurchie_davidson_R_term(tx: int, ty: int, tz: int, n: int = 0) -> str:
    assert tx >= 0 and ty >= 0 and tz >= 0
    assert n >= 0
    term = recur_mcmurchie_davidson_R_term(tx, ty, tz, n)
    assert term is not None

    return simplify(term)
