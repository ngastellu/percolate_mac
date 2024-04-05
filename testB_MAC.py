#!/usr/bin/env python

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from qcnico.qchemMAC import AO_gammas, MO_gammas
from deploy_percolate import setup_hopping_sites_gridMOs
from percolate import diff_arrs, avg_nb_neighbours, pair_inds
from os import path

rCC = 1.8
gamma = 0.1
kB = 8.617333262 #eV / K

datadir = '/Users/nico/Desktop/simulation_outputs/percolation/10x10/'
arpackdir = path.join(datadir,'Hao_ARPACK')
nn = 10

pos, _ = read_xsf(datadir + f'structures/bigMAC-{nn}_relaxed.xsf')
pos = remove_dangling_carbons(pos, rCC)
N = pos.shape[0]

print('Loading H...')
hvals = np.load(path.join(arpackdir,'hvals', f'hvals-{nn}.npy'))
ii = np.load(path.join(arpackdir,'inds', f'ii-{nn}.npy'))
jj = np.load(path.join(arpackdir,'inds', f'jj-{nn}.npy'))
H = np.zeros((N,N))
H[ii-1,jj-1] = hvals
H += H.T
print('Done!\n')

print('Diagonalising...')
energies, M = eigh(H) 
print('Done!\n')

print('Getting AO gammas...')
agL, agR = AO_gammas(pos,gamma)
print('Done!\n')

print('Getting MO gammas...')
gamL, gamR = MO_gammas(M, agL, agR,return_diag=True)
print('Done!\n')

M = M[:,:100]
energies = energies[100:]
gamL = gamL[100:]
gamR = gamR[100:]


rr, ee, *_ = setup_hopping_sites_gridMOs(pos, energies, M, gamL, gamR, nbins=10, save_centers=False)
nsites = ee.shape[0]

eF = 0.5 * (energies[N//2 -1] + energies[N//2])
edArr, rdArr = diff_arrs(ee, rr, a0=30, eF=eF, E=np.array([0.0,0.0]))

nMOs = M.shape[1]
egrid = energies

temps = [40,400]

Bs = []
ds = []

for T in temps:
    dists = rdArr + (edArr/(kB*T))
    dmat = np.zeros((nsites,nsites))
    ij = np.vstack(pair_inds(np.arange(nsites),nsites)).T
    print(ij)
    for IJ, d in zip(ij, dists):
        print(IJ)
        I,J = IJ
        dmat[I,J] = d
        dmat[I,J] = d
    dgrid = np.unique(dists)
    ds.append(dgrid)
    B = avg_nb_neighbours(energies, dmat, egrid, dgrid)
    Bs.append(B)

plt.plot(ds[0], B[0][0,:],'b-',lw=0.8)
plt.plot(ds[1], B[1][0,:],'r-',lw=0.8)
plt.show()

plt.plot(energies, B[0][:,0],'b-',lw=0.8)
plt.plot(energies, B[1][:,0],'r-',lw=0.8)
plt.show()