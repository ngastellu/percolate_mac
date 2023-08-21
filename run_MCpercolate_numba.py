#!/usr/bin/env python

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from percolate import LR_sites_from_MOgams
from monte_carlo import MACHopSites, kMarcus_gu, dipole_coupling
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons


@njit(parallel=True)
def run_MCpercolate(pos, M, MO_energies, sites_data, MO_gams, Js, temps, E, nloops):

    hopsys = MACHopSites(pos,M,MO_energies, sites_data, MO_gams)
    ts = np.ones((nloops, temps.shape[0]), dtype='float') * -1
    for n in prange(nloops):
        print(f"Loop {n}")
        for k in prange(temps.shape[0]):
            T = temps[k]
            print(f"T = {int(T)} K")
            t, _ =  hopsys.MCpercolate_dipoles(Js,T, E, e_reorg=0.005, return_traj=False, interMO_hops_only=True)
            ts[n,k] = t
        print('\n\n')
    return ts

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
print(centers.shape)
print(site_energies.shape)
print(site_inds.shape)

gamL = np.load(percolate_datadir + f'gamL_40x40-{nsample}.npy')
gamR = np.load(percolate_datadir + f'gamR_40x40-{nsample}.npy')

MO_gams = (gamL, gamR)

pos = remove_dangling_carbons(read_xsf(strucdir + f'bigMAC-{nsample}_relaxed.xsf')[0], 1.8 )

Js = np.load(f"/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/dipole_couplings/Jdip-{nsample}.npy")

temps = np.arange(90,410,10,dtype=np.float64)

dX = np.max(pos[:,0]) - np.min(pos[:,1])
E = np.array([1.0,0]) / dX # Efield corresponding to a voltage drop of 1V accross MAC sample 


ts = run_MCpercolate(pos, M, MO_energies, sites_data, MO_gams, Js, temps, E, 100) 
plt.plot(temps,np.mean(ts,axis=0))
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