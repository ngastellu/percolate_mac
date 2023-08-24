#!/usr/bin/env python

import sys
from os import path
import numpy as np
from monte_carlo import MACHopSites


def get_MCtrajs(hopsys, Js, T, E,inter_MO_only):
    print(E)
    print(inter_MO_only)
    _, traj =  hopsys.MCpercolate_dipoles(Js,T, E, 0.005, True, inter_MO_only)
    return traj

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

# pos = remove_dangling_carbons(read_xsf(strucdir + f'bigMAC-{nsample}_relaxed.xsf')[0], 1.8 )
pos = np.load('pos-150_nodangle.npy')

Js = np.load(f"/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/dipole_couplings/Jdip-{nsample}.npy")

dX = np.max(pos[:,0]) - np.min(pos[:,1])

E = np.array([1.0,0]) / dX # Efield corresponding to a voltage drop of 1V accross MAC sample 

# Jfile = f"Jdip-{nsample}.npy"

# if path.exists(Jfile):
#     Js = np.load(Jfile)
# else:
#     Js = dipole_coupling(M,pos,site_inds)
#     np.save(Jfile, Js)



#Js = np.load(f"/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/dipole_couplings/Jdip-{nsample}.npy")

hopsys = MACHopSites(pos,M, eMOs, sites_data, MO_gams)

temps = np.array([80, 100])

for T in temps:
    print(T)
    traj = get_MCtrajs(hopsys, Js, T, E, True)
    np.save(f"traj-{T}K_nointra.npy", traj)

for T in temps:
    traj = get_MCtrajs(hopsys, Js, T, E, False)
    np.save(f"traj-{T}K.npy", traj)

