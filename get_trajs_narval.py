#!/usr/bin/env python

import sys
from os import path
import numpy as np
from numba import njit, prange
from monte_carlo import MACHopSites, dipole_coupling
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons


def get_MCtrajs(hopsys, Js, T, E,inter_MO_only):

    _, traj =  hopsys.MCpercolate_dipoles(Js,T, E, 0.005, True, inter_MO_only)
    return traj

nsample = int(sys.argv[1])

arpackdir = path.expanduser(f"~/scratch/ArpackMAC/40x40/sample-{nsample}")
M = np.load(path.join(arpackdir,f'MOs_ARPACK_bigMAC-{nsample}.npy'))
eMOs = np.load(path.join(arpackdir,f'eARPACK_bigMAC-{nsample}.npy'))

percolate_datadir = path.expanduser(f"~/scratch/MAC_percolate/explicit_percolation/sample-{nsample}/")

centers = np.load(percolate_datadir + 'cc.npy')
site_energies = np.load(percolate_datadir + 'ee.npy')
site_inds = np.load(percolate_datadir + 'ii.npy')

sites_data = (centers, site_energies, site_inds)

gamL = np.load(percolate_datadir + f'gamL_40x40-{nsample}.npy')
gamR = np.load(percolate_datadir + f'gamR_40x40-{nsample}.npy')

MO_gams = (gamL, gamR)

strucdir = path.expanduser(f"~/scratch/clean_bigMAC/40x40/relax/relaxed_structures/")

pos = remove_dangling_carbons(read_xsf(strucdir + f'bigMAC-{nsample}_relaxed.xsf')[0], 1.8 )

dX = np.max(pos[:,0]) - np.min(pos[:,1])

E = np.array([1.0,0]) / dX # Efield corresponding to a voltage drop of 1V accross MAC sample 

Jfile = f"Jdip-{nsample}.npy"

if path.exists(Jfile):
    Js = np.load(Jfile)
else:
    Js = dipole_coupling(M,pos,site_inds)
    np.save(Jfile, Js)



#Js = np.load(f"/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/dipole_couplings/Jdip-{nsample}.npy")

hopsys = MACHopSites(pos,M, eMOs, sites_data, MO_gams)

temps = np.array([80, 100, 200, 400])

for T in temps:
    traj = get_MCtrajs(hopsys, Js, T, E, True)
    np.save(f"traj-{T}K_nointra.npy", traj)

for T in temps:
    traj = get_MCtrajs(hopsys, Js, T, E, False)
    np.save(f"traj-{T}K.npy", traj)

