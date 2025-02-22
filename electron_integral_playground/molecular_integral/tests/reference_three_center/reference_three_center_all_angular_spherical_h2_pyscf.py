
from pyscf import gto, df
import numpy as np

mol = gto.M()
mol.atom = """
    H 0 0 0
    H 1.0 1.3 1.55
"""
mol.basis = """
    H S
        2.0    0.5
        0.5    1.0
    H P
        2.05    0.5
        0.45    1.0
    H D
        2.15    0.5
        0.40    1.0
    H F
        2.25    0.5
        0.35    1.0
    H G
        2.35    0.5
        0.30    1.0
    H H
        2.45    0.5
        0.25    1.0
    H I
        2.55    0.5
        0.20    1.0
"""
mol.unit = 'bohr'
# mol.cart = True
mol.build()

auxbasis = """
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
auxmol = df.addons.make_auxmol(mol, auxbasis)

J3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1', comp=1)
J3c = J3c.transpose(2,0,1)

np.set_printoptions(linewidth = np.iinfo(np.int32).max, threshold = np.iinfo(np.int32).max, precision = 16, suppress = True)
np.savez_compressed('reference_three_center_all_angular_spherical_h2_data.npz', J3c = J3c)
