#!/usr/bin/env python

from os import path
import numpy as np
import matplotlib.pyplot as plt
from percolate import LR_sites_from_MOgams
from monte_carlo import MACHopSites, kMarcus_gu, dipole_coupling
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons


def run_MCpercolate(pos, M, MO_energies, sites_data, MO_gams, nsample, temps, E, e_reorg):

    hopsys = MACHopSites(pos,M,MO_energies, sites_data, MO_gams)
    # Js = np.load(f"/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/dipole_couplings/Jdip-{nsample}.npy")
    Jfile = f"/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/dipole_couplings/Jdip-{nsample}.npy"

    if path.exists(Jfile):
        Js = np.load(Jfile)
    else:
        Js = dipole_coupling(M,pos,site_inds)
        np.save(Jfile, Js)

    ts = np.zeros_like(temps) 
    for k, T in enumerate(temps):
        t, traj =  hopsys.MCpercolate_dipoles(Js,T,E, e_reorg, return_traj=True)
        ts[k] = t
        print(ts[k])
    return ts, traj

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
np.save(f'pos-{nsample}_nodangle.npy', pos)

temps = np.arange(100,300,100,dtype=np.float64)

dX = np.max(pos[:,0]) - np.min(pos[:,1])

E = np.array([1.0,0]) / dX # Efield corresponding to a voltage drop of 1V accross MAC sample 
e_reorg = 0.005

for iii in range(10):

    ts = run_MCpercolate(pos, M, MO_energies, sites_data, MO_gams,nsample, temps, E, e_reorg) 

plt.plot(temps,ts)
plt.show()

# Js = dipole_coupling(M,pos,site_inds)
# np.save(f"/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/dipole_couplings/Jdip-{nsample}.npy", Js)



# # dE = site_energies[None,:] - site_energies[:,None]
# # #dE[np.abs(dE) == 0] = 1e-6

# K = kMarcus_gu(site_energies,centers,0.01,Js,100,np.array([1,0]))

# L,R = LR_sites_from_MOgams(gamL, gamR, site_inds, return_ndarray=True)

# for i in L[:5]:
#     plt.plot(K[i,:],label=f"{i}")
#     print(K[i,:])
# plt.legend()
# plt.show()