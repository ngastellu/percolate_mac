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

# rCC = 1.8
# gamma = 0.1
kB = 8.617333262 #eV / K

# datadir = '/Users/nico/Desktop/simulation_outputs/percolation/10x10/'
# arpackdir = path.join(datadir,'Hao_ARPACK')
# nn = 10

# pos, _ = read_xsf(datadir + f'structures/bigMAC-{nn}_relaxed.xsf')
# pos = remove_dangling_carbons(pos, rCC)
# N = pos.shape[0]

# print('Loading H...')
# hvals = np.load(path.join(arpackdir,'hvals', f'hvals-{nn}.npy'))
# ii = np.load(path.join(arpackdir,'inds', f'ii-{nn}.npy'))
# jj = np.load(path.join(arpackdir,'inds', f'jj-{nn}.npy'))
# H = np.zeros((N,N))
# H[ii-1,jj-1] = hvals
# H += H.T
# print('Done!\n')

# print('Diagonalising...')
# energies, M = eigh(H) 
# print('Done!\n')

# print('eshape 1: = ', energies.shape[0])

# print('Getting AO gammas...')
# agL, agR = AO_gammas(pos,gamma)
# print('Done!\n')

# print('Getting MO gammas...')
# gamL, gamR = MO_gammas(M, agL, agR,return_diag=True)
# print('Done!\n')

# M = M[:,:10]
# energies = energies[:10]
# gamL = gamL[:10]
# gamR = gamR[:10]

# print('eshape 2: = ', energies.shape[0])

# rr, ee, *_ = setup_hopping_sites_gridMOs(pos, energies, M, gamL, gamR, nbins=10, save_centers=False)
# nsites = ee.shape[0]

# print("N = ", N)
# print("N // 2 = ", N//2)

# # eF = 0.5 * (energies[N//2 -1] + energies[N//2])
# eF = 0.0


np.random.seed(42069)

nMOs = 5
nsites = nMOs

# ee = np.random.rand(nMOs) * 2 - 1
ee = np.zeros(nMOs)
# ee[np.arange(nMOs) % 2 == 1] = 1
rr = np.random.rand(nMOs,2) * 5
print('*** Positions ***')
print(rr)
print('********')
eF = 0.0
edArr, rdArr = diff_arrs(ee, rr, a0=2, eF=eF, E=np.array([0.0,0.0]))
print(edArr)


# nMOs = M.shape[1]
# egrid = energies
energies = ee
egrid = np.sort(energies)

# temps = [40,400]

# Bs = [] 
# ds = []

# for T in temps:
T = 100
dists = rdArr + (edArr/(kB*T))
print(dists)
dmat = np.zeros((nsites,nsites))
ij = np.vstack(pair_inds(np.arange(dists.shape[0]),nsites)).T
print(ij)
for IJ, d in zip(ij, dists):
    print(IJ)
    I,J = IJ
    print(I)
    print(J)
    dmat[I,J] = d
    dmat[J,I] = d
    # print(dmat[I,J])
    # print(dmat[J,I])

# dmat += dmat.T
for i in range(nsites): #remove diagonal elements from the calc
    dmat[i,i] = np.max(dists)+100 

print(dmat)
print('\n----------------------\n')
dmat2 = np.linalg.norm(rr[:,None,:] - rr,axis=2)
for i in range(nsites): #remove diagonal elements from the calc
    dmat2[i,i] = np.max(dists)+100 
print(dmat2)
print('\n')


print(dmat == dmat2)

dgrid = np.unique(dists)[:20]
print(dgrid)
    # ds.append(dgrid)
B = avg_nb_neighbours(energies, dmat, egrid, dgrid)
    # Bs.append(B)



plt.scatter(*rr.T,c=ee)
plt.scatter(*rr[0,:],c='none',edgecolors='r', s=100, linewidths=0.8)

neighbs0 = np.argmin(dmat[0,:])
plt.scatter(*rr[neighbs0,:].T,c='none',edgecolors='b', s=100, linewidths=0.8)
plt.show()

print(np.all(dmat - dmat.T == 0))

# plt.plot(ds[0], Bs[0][0,:],'-o',lw=0.8)
# plt.plot(ds[1], Bs[1][0,:],'-o',lw=0.8)
# plt.show()

# plt.plot(egrid, Bs[0][:-1,0],'-o',lw=0.8)
# plt.plot(egrid, Bs[1][:-1,0],'-o',lw=0.8)
# plt.show()