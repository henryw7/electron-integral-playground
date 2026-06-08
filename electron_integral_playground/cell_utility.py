
import numpy as np

from electron_integral_playground.data_structure import Cell

def get_cell_from_lattice_constant(a : float, b : float, c : float, alpha : float, beta : float, gamma : float, dimension : int = 3) -> Cell:
    """
    Expect a,b,c in Bohr, alpha,beta,gamma in radian.
    """
    assert dimension in (1,2,3)
    assert dimension == 3, "2D or 1D periodic system not supported"

    R = np.zeros((3,3), order = "C")
    from numpy import sin, cos, sqrt
    R[0,0] = a
    R[1,0] = b * cos(gamma)
    R[1,1] = b * sin(gamma)
    R[2,0] = c * cos(beta)
    R[2,1] = c * (cos(alpha) - cos(beta)*cos(gamma)) / sin(gamma)
    V = a * b * c * sqrt(1 - cos(alpha)*cos(alpha) - cos(beta)*cos(beta) - cos(gamma)*cos(gamma) + 2*cos(alpha)*cos(beta)*cos(gamma))
    R[2,2] = V / (a * b * sin(gamma))

    return Cell(
        dimension = dimension,
        lattice_vectors = R,
    )


