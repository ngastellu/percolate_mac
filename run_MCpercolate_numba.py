#!/usr/bin/env python

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from monte_carlo import MACHopSites


@njit(parallel=True)
def run_MCpercolate(pos, M, MO_energies, sites_data, MO_gams, temps, E, nloops, ts):

    hopsys = MACHopSites(pos,M,MO_energies, sites_data, MO_gams)
    a0 = 30
    for n in prange(nloops):
        print(f"Loop {n}")
        for k in prange(temps.shape[0]):
            T = temps[k]
            print(f"T = {int(T)} K")
            t, _ =  hopsys.MCpercolate_MA(T, E, a0, return_traj=False, interMO_hops_only=False)
            ts[n,k] = t
        print('\n\n')
    return ts


nsample = 150

percolate_datadir = f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/sample-{nsample}/'
M = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/MOs_ARPACK/MOs_ARPACK_bigMAC-{nsample}.npy')
eMOs = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/eARPACK/eARPACK_bigMAC-{nsample}.npy')
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

pos = np.load(strucdir + f'no_dangle/pos-{nsample}_nodangle.npy')

temps = np.arange(90,110,10,dtype=np.float64)
nloops = 10
ts = np.ones((nloops, temps.shape[0]), dtype='float') * -1

# dX = np.max(pos[:,0]) - np.min(pos[:,1])
# E = np.array([1.0,0]) / dX # Efield corresponding to a voltage drop of 1V accross MAC sample 

E = np.zeros(2)

ts = np.ones((nloops, temps.shape[0]), dtype='float') * -1
ts = run_MCpercolate(pos, M, eMOs, sites_data, MO_gams, temps, E, nloops, ts) 

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