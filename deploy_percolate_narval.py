#!/usr/bin/env python

import sys
import pickle
from os import path
import numpy as np
from mpi4py import MPI
import qcnico.qchemMAC as qcm
from qcnico.coords_io import read_xsf
from percolate import diff_arrs, percolate


sample_index = int(sys.argv[1])



# ******* 0: Partition tasks over different cores *******
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

all_Ts = np.arange(0,1050,50)
ops_per_rank = all_Ts.shape[0] // nprocs
if rank == nprocs -1:
    Ts = all_Ts[rank*ops_per_rank:]
else:
    Ts = all_Ts[rank*ops_per_rank:(rank+1)*ops_per_rank]

kB = 8.617e-5

# ******* 1: Load data *******
arpackdir = path.expanduser('~/scratch/ArpackMAC/40x40')
pos_dir = path.expanduser('~/scratch/clean_bigMAC/40x40/relax/relaxed_structures')
sample_dir = f'sample-{sample_index}'


mo_file = f'MOs_ARPACK_bigMAC-{sample_index}.npy'
energy_file = f'eARPACK_bigMAC-{sample_index}.npy'

mo_path = path.join(arpackdir,sample_dir,mo_file)
energy_path = path.join(arpackdir,sample_dir,energy_file)
pos_path = path.join(pos_dir,sample_dir,f'bigMAC-{sample_index}_relaxed.xsf')

energies = np.load(energy_path)
M =  np.load(mo_path)
pos, _ = read_xsf(pos_path)  


# ******* 2: Get gammas *******
ga = 0.1 #edge atome-lead coupling in eV
print("Computing AO gammas...")
agaL, agaR = qcm.AO_gammas(pos,ga)
print("Computing MO gammas...")
gamL, gamR = qcm.MO_gammas(M,agaL, agaR, return_diag=True)
np.save(f'gamL_40x40-{sample_index}.npy',gamL)
np.save(f'gamR_40x40-{sample_index}.npy',gamR)

# gamL = np.load(f'gamL_40x40-{sample_index}.npy')
# gamR = np.load(f'gamR_40x40-{sample_index}.npy')


# ******* 3: Define strongly-coupled MOs *******
tolscal = 3.0
gamL_tol = np.mean(gamL) + tolscal*np.std(gamL)
gamR_tol = np.mean(gamR) + tolscal*np.std(gamR)

biggaL_inds = (gamL > gamL_tol).nonzero()[0]
biggaR_inds = (gamR > gamR_tol).nonzero()[0]


# ******* 4: Get a sense of the distance distribution *******
coms = qcm.MO_com(pos, M)

for T in Ts:
    edArr, rdArr = diff_arrs(energies, coms, a0=1, eF=0)
    # ******* 5: Get spanning cluster *******
    conduction_clusters, dcrit, A = percolate(energies, pos, M, gamL_tol=gamL_tol,gamR_tol=gamR_tol, return_adjmat=True, distance='logMA',MOgams=(gamL, gamR), dArrs=(edArr,rdArr))

    with open(f'out_percolate-{T}K.pkl', 'wb') as fo:
        pickle.dump((conduction_clusters,dcrit,A), fo)
