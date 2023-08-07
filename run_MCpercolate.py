#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from monte_carlo import MACHopSites

nsample = 150

percolate_datadir = f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/sample-{nsample}/'
M = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/MOs_ARPACK/MOs_ARPACK_bigMAC-{nsample}.npy')
MO_energies = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/eARPACK/eARPACK_bigMAC-{nsample}.npy')
print(M.shape)

centers = np.load(percolate_datadir + 'cc.npy')
site_energies = np.load(percolate_datadir + 'ee.npy')
site_inds = np.load(percolate_datadir + 'ii.npy')

sites_data = (centers, site_energies, site_inds)

gamL_sites = np.load(percolate_datadir + f'gamL_40x40-{nsample}.npy')
gamR_sites = np.load(percolate_datadir + f'gamR_40x40-{nsample}.npy')

edge_sites = (gamL_sites, gamR_sites)

pos = 1 #unnecessary here because the hopping sites have already been computed

hopsys = MACHopSites(pos,M,MO_energies, sites_data, edge_sites)
vdos = 1


temps = np.arange(100,500,100)
ts = np.zeros_like(temps)
for k, T in enumerate(temps):
    ts[k] = hopsys.MCpercolate(T, vdos)

plt.plot(temps,ts)
plt.show()


