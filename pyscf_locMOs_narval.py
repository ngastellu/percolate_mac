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

strucfile = path.expanduser(f'~/scratch/clean_bigMAC/40x40/relax/no_PBC/relaxed_structures/bigMAC-{nn}_relaxed.xsf')
MOfile = path.expanduser(f'~/scratch/ArpackMAC/40x40/sample-{nn}/MOs_ARPACK_bigMAC-{nn}.npy')

pos, _ = read_xsf(strucfile)
pos = remove_dangling_carbons(pos,rCC)
M = np.load(MOfile)

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
loc_orb = lo.Boys(mol, M).kernel()
end = perf_counter()
print(f"Done localising! [{end-start}s]")

np.save('boyz_MOs_102x102.npy', loc_orb)