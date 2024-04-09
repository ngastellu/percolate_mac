#!/usr/bin/env python

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from qcnico.qchemMAC import AO_gammas, MO_gammas
from deploy_percolate import setup_hopping_sites_gridMOs
from percolate import diff_arrs, avg_nb_neighbours, pair_inds, diff_arrs_w_inds
from os import path
from time import perf_counter

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

diag_start = perf_counter()
print('Diagonalising...')
energies, M = eigh(H) 
diag_end = perf_counter()
print(f'Done! [{diag_end - diag_start} seconds]\n')

egrid = np.linspace(np.min(energies) - 0.001, np.max(energies) + 0.001, 1000)

print('eshape 1: = ', energies.shape[0])

print('Getting AO gammas...')
ag_start = perf_counter()
agL, agR = AO_gammas(pos,gamma)
ag_end = perf_counter()
print(f'Done! [{ag_end - ag_start} seconds]\n')
print('Done!\n')

print('Getting MO gammas...')
mg_start = perf_counter()
gamL, gamR = MO_gammas(M, agL, agR,return_diag=True)
mg_end = perf_counter()
print(f'Done! [{mg_end - mg_start} seconds]\n')

# M = M[:,:10]
# energies = energies[:10]
# gamL = gamL[:10]
# gamR = gamR[:10]

print('eshape 2: = ', energies.shape[0])

print("Getting hopping sites...")
sites_start = perf_counter()
rr, ee, *_ = setup_hopping_sites_gridMOs(pos, energies, M, gamL, gamR, nbins=10, save_centers=False)
sites_end = perf_counter()
print(f'Done! [{sites_end - sites_start} seconds]\n')

nsites = ee.shape[0]

print("N = ", N)
print("N // 2 = ", N//2)

eF = 0.5 * (energies[N//2 -1] + energies[N//2])

print("Getting energy and position difference arrays...")
d_start = perf_counter()
edArr, rdArr, ij = diff_arrs_w_inds(ee, rr, a0=2, eF=eF, E=np.array([0.0,0.0]))
d_end = perf_counter()
print(f'Done! [{d_end - d_start} seconds]\n')

# print("Getting pair inds...")
# i_start = perf_counter()
# ij = np.vstack(pair_inds(np.arange(rdArr.shape[0]),nsites)).T
# i_end = perf_counter()
# print(f'Done! [{i_end - i_start} seconds]\n')
temps = [40,400]

Bs = [] 
ds = []

for T in temps:
    print(f"\n***** T = {T}K *****")
    dists = rdArr + (edArr/(kB*T))
    dgrid = np.linspace(np.min(dists)-0.1, np.max(dgrid)+0.1, 1000)
    print(dists)
    dmat = np.zeros((nsites,nsites))
    # print(ij)
    for IJ, d in zip(ij, dists):
        I,J = IJ
        dmat[I,J] = d
        dmat[J,I] = d

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
    ds.append(dgrid)
    print("Computing B...")
    b_start = perf_counter()
    B = avg_nb_neighbours(energies, dmat, egrid, dgrid)
    b_end = perf_counter()
    print(f'Done! [{b_end - b_start} seconds]\n')
    np.save(f'B-{T}K.npy', B)
    np.save(f'dgrid-{T}K.npy', dgrid)