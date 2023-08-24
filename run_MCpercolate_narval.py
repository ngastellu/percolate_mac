#!/usr/bin/env python

import sys
from os import path
import numpy as np
from numba import njit, prange
from monte_carlo import MACHopSites, dipole_coupling
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

# dX = np.max(pos[:,0]) - np.min(pos[:,1])

# E = np.array([1.0,0]) / dX # Efield corresponding to a voltage drop of 1V accross MAC sample 

Jfile = f"Jdip-{nsample}.npy"

# if path.exists(Jfile):
#     Js = np.load(Jfile)
# else:
#     Js = dipole_coupling(M,pos,site_inds)
#     np.save(Jfile, Js)

print("Calculating J...", flush=True)
Js = dipole_coupling(M,pos,site_inds)
print('Done!', flush=True)
print("Saving J to disk...", flush=True)
np.save(Jfile, Js)
print('Done!', flush=True)

#Js = np.load(f"/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/dipole_couplings/Jdip-{nsample}.npy")


temps = np.arange(70,505,5,dtype=np.float64)
ts = run_MCpercolate(pos, M, eMOs, sites_data, MO_gams, Js, temps, np.zeros(2), 1000) 

np.save(f'dipole_perc_times-{nsample}_zero_field.npy', ts)
