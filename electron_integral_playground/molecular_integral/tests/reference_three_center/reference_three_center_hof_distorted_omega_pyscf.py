
from pyscf import gto, df
import numpy as np

mol = gto.M()
mol.atom = """
        H -1 0.1 0
        O 0 0 0
        F 0.5 0.6 2.7
"""
mol.basis = "sto-3g"
omega = 0.2
mol.set_range_coulomb(omega)
mol.build()

auxmol = df.addons.make_auxmol(mol, auxbasis = "def2-svp-ri")

J3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1', comp=1)
J3c = J3c.transpose(2,0,1)

np.set_printoptions(linewidth = np.iinfo(np.int32).max, threshold = np.iinfo(np.int32).max, precision = 16, suppress = True)
np.savez_compressed('reference_three_center_hof_distorted_omega_data.npz', J3c = J3c)
