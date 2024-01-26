import numpy
import pyscf
from pyscf import lo


"""Beginning of this example is copied from https://arxiv.org/pdf/2002.12531.pdf"""

mol = pyscf.M()
n = 100
mol.nelectron = n

# Define model Hamiltonian: tight binding on a ring
h1 = numpy.zeros((n, n))
for i in range(n-1):
    h1[i, i+1] = h1[i+1, i] = -1.
    h1[n- 1, 0] = h1[0, n-1] = -1.

# Build the 2-electron interaction tensor starting from a random 3-index tensor.
tensor = numpy.random.rand(2, n, n)
tensor = tensor + tensor.transpose(0, 2, 1)
eri = numpy.einsum('xpq,xrs->pqrs', tensor, tensor)

# SCF for the custom Hamiltonian
mf = mol.HF()
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: numpy.eye(n)

print(mol.basis)

loc_orb = lo.Boys(mol)
print(loc_orb.mo_coeff)