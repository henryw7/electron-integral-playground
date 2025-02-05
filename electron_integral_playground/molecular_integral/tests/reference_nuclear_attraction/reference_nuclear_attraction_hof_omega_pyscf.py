
from pyscf import gto
import numpy as np

mol = gto.M(
    atom = """
        H -1 0.1 0
        O 0 0 0
        F 0.5 0.6 0.7
    """,
    basis = "def2-svp",
)
omega = 0.3
mol.set_range_coulomb(omega)
mol.build()

V = mol.intor("int1e_grids", grids = mol.atom_coords())
V = V.transpose(1,2,0)

np.savetxt('reference_nuclear_attraction_hof_omega_data.txt', V.flatten())

