#!/usr/bin/env python

import sys
from os import path
import numpy as np
from numba import njit, prange
from monte_carlo import MACHopSites
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons


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
#
#dX = np.max(pos[:,0]) - np.min(pos[:,1])
#
#E = np.array([1.0,0]) / (1000*dX) # Efield corresponding to a voltage drop of 1V accross MAC sample 
#
#Jfile = f"Jdip-{nsample}.npy"

E = np.zeros(2)

temps = np.arange(70,505,5,dtype=np.float64)
nloops = 1000


ts = np.ones((nloops, temps.shape[0]), dtype='float') * -1
ts = run_MCpercolate(pos, M, eMOs, sites_data, MO_gams, temps, E, nloops, ts) 

np.save(f'MA_perc_times-{nsample}_zero_field.npy', ts)
