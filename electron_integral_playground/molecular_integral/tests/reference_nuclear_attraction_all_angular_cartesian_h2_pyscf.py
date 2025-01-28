
from pyscf import gto
import numpy as np

mol = gto.M(
    atom = """
        H 0 0 0
        H 1.0 1.3 1.55
    """,
    basis = """
        H S
            1.0    1.0
            0.5    1.0
        H P
            1.2    1.0
            0.6    1.0
        H D
            1.4    1.0
            0.7    1.0
        H F
            1.6    1.0
            0.8    1.0
        H G
            1.8    1.0
            0.9    1.0
        H H
            2.0    1.0
            1.0    1.0
        H I
            2.2    1.0
            1.1    1.0
    """,
    unit = 'bohr',
    cart = True,
)
mol.build()

charge_position = np.array([
    [ 0, 0, 0, ],
    [ 1.0, 1.3, 1.55, ],
    [ 0.5, 0.65, 0.775, ],
    [ 10, 0, 0, ],
    [ 10, 2, 0, ],
    [ 50, 2, 0, ],
    [ 100, 2, 0, ],
    [ 500, 2, 0, ],
    [ 1, 0, 0, ],
    [ 0, 1, 0, ],
    [ 0, 0, 1, ],
    [ 1, 1, 1, ],
    [ 0, 0, 0.001, ],
    [ -3, 0, 0, ],
])

V = mol.intor("int1e_grids", grids = charge_position)
V = V.transpose(1,2,0)

S = mol.intor("int1e_ovlp")
for i in range(S.shape[0]):
    if abs(S[i,i] - 1.0) > 1e-14:
        inverse_sqrt_S_diagonal = 1.0 / np.sqrt(S[i,i])
        V[i, :, :] *= inverse_sqrt_S_diagonal
        V[:, i, :] *= inverse_sqrt_S_diagonal

np.set_printoptions(linewidth = np.iinfo(np.int32).max, threshold = np.iinfo(np.int32).max, precision = 16, suppress = True)
np.savetxt('reference_nuclear_attraction_all_angular_cartesian_h2_data.txt', V.flatten())
