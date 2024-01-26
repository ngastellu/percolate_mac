#!/usr/bin/env python

import numpy as np
from time import perf_counter
from pyscf import gto, lo
from qcnico.qcffpi_io import read_MO_file


pos, M = read_MO_file("MOs_pCNN_MAC_102x102.dat")
print(pos.shape)
print(M.shape)

N = pos.shape[0]

print("Building mol object...")
start = perf_counter()
mol = gto.Mole()
mol.atom = [['C', tuple(r)] for r in pos]
mol.basis = {'C': gto.basis.parse("""
C    S
2.9412494   0.15591627
0.6834831   0.60768372
0.2222899   0.39195739
""")}
mol.build()
end = perf_counter()
print(f"Done building! [{end-start}s]")

print(mol.basis)

# mf = scf.hf.SCF(mol).build()
mol.mo_coeff = M


N = pos.shape[0]
#print(lo.orth.orth_ao(mol,method='lowdin',pre_orth_ao=np.eye(N),s=np.eye(N)))

print("Localising orbitals...")
start = perf_counter()
loc_orb = lo.Boys(mol, M[:,int(N/2):100+int(N/2)]).kernel()
end = perf_counter()
print(f"Done localising! [{end-start}s]")

np.save('boyz_MOs_102x102.npy', loc_orb)