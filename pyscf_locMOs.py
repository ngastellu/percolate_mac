#!/usr/bin/env python

import sys
from os import path
import numpy as np
from time import perf_counter
from pyscf import gto, lo
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons



nn = sys.argv[1]
rCC = 1.8

strucfile = f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/structures/bigMAC-{nn}_relaxed.xsf'
MOfile = f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/MOs_ARPACK/MOs_ARPACK_bigMAC-{nn}.npy'
pos, _ = read_xsf(strucfile)
pos = remove_dangling_carbons(pos,rCC)
M = np.load(MOfile)
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
loc_orb = lo.Boys(mol).kernel(mo_coeff=M[:,:3])
end = perf_counter()
print(f"Done localising! [{end-start}s]")

np.save(f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/boyz_ARPACK/boyz_bigMAC-{nn}.npy', loc_orb)