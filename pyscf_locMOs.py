#!/usr/bin/env python

from os import path
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, lo, scf
from pyscf.tools import molden
# from qcnico.coords_io import read_xsf
from qcnico.qcffpi_io import read_MO_file
from qcnico.qcplots import plot_MO


pos, M = read_MO_file("/Users/nico/Desktop/simulation_outputs/qcffpi_data/MO_coefs/MOs_pCNN_MAC_102x102.dat")
print(pos.shape)
print(M.shape)

N = pos.shape[0]

mol = gto.Mole()
mol.atom = [['C', tuple(r)] for r in pos]
mol.basis = {'C': gto.basis.parse("""
C    S
2.9412494   0.15591627
0.6834831   0.60768372
0.2222899   0.39195739
""")}
mol.build()

print(mol.basis)

# mf = scf.hf.SCF(mol).build()
mol.mo_coeff = M



#print(lo.orth.orth_ao(mol,method='lowdin',pre_orth_ao=np.eye(N),s=np.eye(N)))
loc_orb = lo.Boys(mol, M[:,190:200]).kernel()
print(loc_orb)

for n in range(190,195):
    plot_MO(pos, M, n)
    plot_MO(pos, loc_orb, n-190, cmap='viridis')




# mol = gto.M(
#     atom = '''
# C    0.000000000000     1.398696930758     0.000000000000
# C    0.000000000000    -1.398696930758     0.000000000000
# C    1.211265339156     0.699329968382     0.000000000000
# C    1.211265339156    -0.699329968382     0.000000000000
# C   -1.211265339156     0.699329968382     0.000000000000
# C   -1.211265339156    -0.699329968382     0.000000000000
# H    0.000000000000     2.491406946734     0.000000000000
# H    0.000000000000    -2.491406946734     0.000000000000
# H    2.157597486829     1.245660462400     0.000000000000
# H    2.157597486829    -1.245660462400     0.000000000000
# H   -2.157597486829     1.245660462400     0.000000000000
# H   -2.157597486829    -1.245660462400     0.000000000000''',
#     basis = '6-31g')
# mf = scf.RHF(mol).run()


# mf = scf.RHF(mol).run()
# print(mf.mo_coeff.shape)
# print(mf.mo_coeff.dtype)
# M = mf.mo_coeff
# M2 = M.T @ M
# plt.imshow(np.abs(M.T @ M) > 1e-6)
# plt.show()

# print(np.mean(M2.diagonal()))
# print( ( np.mean(M2[np.triu_indices(M2.shape[0],1)]) + np.mean(M2[np.tril_indices(M2.shape[0],-1)]))*0.5)


# loc_orb = lo.Boys(mol, mf.mo_coeff[:,:3]).kernel()
# print(loc_orb)