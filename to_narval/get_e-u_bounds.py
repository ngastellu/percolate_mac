#!/usr/bin/env python

import sys
from time import perf_counter
from os import path
import numpy as np
from scipy.linalg import eigh
from percolate import diff_arrs
from deploy_percolate import setup_hopping_sites_gridMOs
from MOs2sites import generate_site_list
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from qcnico import qchemMAC as qcm


nn = int(sys.argv[1])

kB = 8.617e-5 #eV/K
rCC = 1.8 #angstroms
gamma = 0.1 #eV
tolscal_gamma = 3.0

Tmin = 40
Tmax = 430

posdir = path.expanduser('~/scratch/clean_bigMAC/20x20/relax/relaxed_structures/')
edir = path.expanduser('~/scratch/ArpackMAC/20x20/dense_tb_eigvals/')
Mdir = path.expanduser('~/scratch/ArpackMAC/20x20/dense_tb_eigvecs/')



pos, _ = read_xsf(posdir + f'bigMAC-{nn}_relaxed.xsf')
pos = remove_dangling_carbons(pos, rCC)
N = pos.shape[0]

M = np.load(Mdir + f'eigvecs-{nn}.npy')
energies = np.load(edir + f'eigvals-{nn}.npy')

print('Getting AO gammas...')
ag_start = perf_counter()
agL, agR = qcm.AO_gammas(pos,gamma)
ag_end = perf_counter()
print(f'Done! [{ag_end - ag_start} seconds]\n')
print('Done!\n')

print('Getting MO gammas...')
mg_start = perf_counter()
gamL, gamR = qcm.MO_gammas(M, agL, agR,return_diag=True)
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
energies -= eF

print("Getting energy and position difference arrays...")
d_start = perf_counter()
dE, dR = diff_arrs(ee, rr, a0=30, eF=0, E=np.array([0.0,0.0]))
d_end = perf_counter()
print(f'Done! [{d_end - d_start} seconds]\n')



print('Getting max/min dists...')
dists = (dE/(kB*Tmin)) + dR

dsave = np.zeros(2)
dsave[0] = np.min(dists)
dsave[1]= np.max(dists)

esave = np.zeros(2)
esave[0] = np.min(energies)
esave[1]= np.max(energies)

np.save(f'extreme_dists-{nn}.npy', dsave)
print('Done!\n')