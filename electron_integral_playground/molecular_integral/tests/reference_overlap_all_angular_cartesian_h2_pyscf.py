
from pyscf import gto, scf
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

np.set_printoptions(linewidth = np.iinfo(np.int32).max, threshold = np.iinfo(np.int32).max, precision = 16, suppress = True)
S = mol.intor("int1e_ovlp")

for i in range(S.shape[0]):
    if abs(S[i,i] - 1.0) > 1e-14:
        inverse_sqrt_S_diagonal = 1.0 / np.sqrt(S[i,i])
        S[i, :] *= inverse_sqrt_S_diagonal
        S[:, i] *= inverse_sqrt_S_diagonal

np.savetxt('reference_overlap_all_angular_cartesian_h2_data.txt', S)

# mf = scf.RHF(mol)
# mf.kernel()
# print(mf.get_ovlp())
