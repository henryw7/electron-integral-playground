
from pyscf import gto
import numpy as np

mol = gto.M(
    atom = """
        H -1 0.1 0
        O 0 0 0
        F 0.5 0.6 0.7
    """,
    basis = "def2-svp"
)
mol.build()

S = mol.intor("int1e_ovlp")

np.savetxt('reference_overlap_hof.txt', S)

