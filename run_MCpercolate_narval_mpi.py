#!/usr/bin/env python

import sys
from os import path
import numpy as np
import matplotlib.pyplot as plt
from monte_carlo import MACHopSites, dipole_coupling
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from mpi4py import MPI


comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

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

hopsys = MACHopSites(pos,M,eMOs, sites_data, MO_gams)

Js = dipole_coupling(M,pos,site_inds)
np.save(f"Jdip-{nsample}.npy", Js)

#Js = np.load(f"/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/dipole_couplings/Jdip-{nsample}.npy")


all_temps = np.arange(70,510,5,dtype=np.float64)

temps_per_proc = all_temps.shape[0] // nprocs 

if rank < nprocs-1:
      temps = all_temps[rank*temps_per_proc: (rank+1)*temps_per_proc]
else:
      temps = all_temps[rank*temps_per_proc:]

print(f"Process nb. {rank}: working on temps {temps[0]} K --> {temps[-1]} K ({temps.shape[0]} points).")
ts = np.zeros_like(temps) 
nloops = 10

for k, T in enumerate(temps):
    print(f"* * * * * * * * * * {T} * * * * * * * * * *")
    for n in range(nloops):
        print(n)
        ts[k] += hopsys.MCpercolate_dipoles(Js,T,E)/nloops
    print('\n\n')

np.save(f'dipole_perc_times-{nsample}-{rank}.npy', ts)