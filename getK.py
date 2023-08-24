#!/usr/bin/env python

from os import path
import numpy as np
import matplotlib.pyplot as plt
from monte_carlo import kMarcus_njit, dipole_coupling


nsample = 150

percolate_datadir = f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/sample-{nsample}/'
M = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/MOs_ARPACK/MOs_ARPACK_bigMAC-{nsample}.npy')
MO_energies = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/eARPACK/eARPACK_bigMAC-{nsample}.npy')
strucdir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/structures/'
print(M.shape)

centers = np.load(percolate_datadir + 'cc.npy')
site_energies = np.load(percolate_datadir + 'ee.npy')
site_inds = np.load(percolate_datadir + 'ii.npy')

sites_data = (centers, site_energies, site_inds)

gamL = np.load(percolate_datadir + f'gamL_40x40-{nsample}.npy')
gamR = np.load(percolate_datadir + f'gamR_40x40-{nsample}.npy')

MO_gams = (gamL, gamR)

pos = np.load(f'pos-{nsample}_nodangle.npy')

np.save(f'pos-{nsample}_nodangle.npy', pos)

temps = np.arange(100,300,100,dtype=np.float64)

dX = np.max(pos[:,0]) - np.min(pos[:,1])

E = np.array([0.0,0]) / dX # Efield corresponding to a voltage drop of 1V accross MAC sample 
e_reorg = 0.005


Js = dipole_coupling(M,pos,site_inds)
# np.save(f"/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/dipole_couplings/Jdip-{nsample}.npy", Js)

# Js = np.load(f"/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/dipole_couplings/Jdip-{nsample}.npy")

# dE = site_energies[None,:] - site_energies[:,None]
# #dE[np.abs(dE) == 0] = 1e-6



K = kMarcus_njit(site_energies,centers,e_reorg,Js,100,np.zeros(2))


np.save("kMarcus-150.npy", K)