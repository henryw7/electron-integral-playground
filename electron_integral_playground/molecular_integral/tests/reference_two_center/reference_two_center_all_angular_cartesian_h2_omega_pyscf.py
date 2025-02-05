
from pyscf import gto, df
import numpy as np

mol = gto.M()
mol.atom = """
    H 0 0 0
    H 1.0 1.3 1.55
"""
mol.basis = """
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
    H K
        1.9    1.0
        0.9    1.0
    H L
        1.8    1.0
        0.8    1.0
    H M
        1.7    1.0
        0.7    1.0
"""
mol.unit = 'bohr'
mol.cart = True
omega = 0.3
mol.set_range_coulomb(omega)
mol.build()

auxmol = df.addons.make_auxmol(mol, mol.basis)
J2c = auxmol.intor('int2c2e', aosym='s1', comp=1)

S = mol.intor("int1e_ovlp")
for i in range(S.shape[0]):
    if abs(S[i,i] - 1.0) > 1e-14:
        inverse_sqrt_S_diagonal = 1.0 / np.sqrt(S[i,i])
        J2c[i, :] *= inverse_sqrt_S_diagonal
        J2c[:, i] *= inverse_sqrt_S_diagonal

np.set_printoptions(linewidth = np.iinfo(np.int32).max, threshold = np.iinfo(np.int32).max, precision = 16, suppress = True)
np.savetxt('reference_two_center_all_angular_cartesian_h2_omega_data.txt', J2c)
