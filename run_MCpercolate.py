#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from percolate import LR_sites_from_MOgams
from monte_carlo import MACHopSites, kMarcus_gu, dipole_coupling
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons


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

pos = remove_dangling_carbons(read_xsf(strucdir + f'bigMAC-{nsample}_relaxed.xsf')[0], 1.8 )

hopsys = MACHopSites(pos,M,MO_energies, sites_data, MO_gams)
vdos = np.array([1])

# Js = dipole_coupling(M,pos,site_inds)
# np.save(f"/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/dipole_couplings/Jdip-{nsample}.npy", Js)

Js = np.load(f"/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/dipole_couplings/Jdip-{nsample}.npy")

temps = np.arange(100,500,100,dtype=np.float64)
ts = np.zeros_like(temps) 
nsamples = 5
for k, T in enumerate(temps):
    ts[k] += hopsys.MCpercolate_dipoles(Js,T)
    print(ts[k])

plt.plot(temps,ts)
plt.show()

# dE = site_energies[None,:] - site_energies[:,None]
# #dE[np.abs(dE) == 0] = 1e-6

# Js = dipole_coupling(M,pos,site_inds)
K = kMarcus_gu(site_energies,centers,0.1,Js,100,np.array([1,0]))

L,R = LR_sites_from_MOgams(gamL, gamR, site_inds, return_ndarray=True)

for i in L[:5]:
    plt.plot(K[i,:],label=f"{i}")
    print(K[i,:])
plt.legend()
plt.show()