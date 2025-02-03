
from pyscf import gto, df
import numpy as np

mol = gto.M()
mol.atom = """
    H -1 0.1 0
    O 0 0 0
    F 0.5 0.6 0.7
"""
mol.basis = "sto-3g"
mol.build()

auxmol = df.addons.make_auxmol(mol, auxbasis = "def2-svp-ri")
J2c = auxmol.intor('int2c2e', aosym='s1', comp=1)

np.savetxt('reference_two_center_hof_data.txt', J2c)
