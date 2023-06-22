#!/usr/bin/env python

import sys
import pickle
from os import path
import numpy as np
#from mpi4py import MPI
import qcnico.qchemMAC as qcm
from qcnico.coords_io import read_xsf
from percolate import diff_arrs, percolate, plot_cluster, generate_site_list


sample_index = 99 #int(sys.argv[1])



# ******* 0: Partition tasks over different cores *******
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# nprocs = comm.Get_size()

# all_Ts = np.arange(0,1050,50)
# ops_per_rank = all_Ts.shape[0] // nprocs
# if rank == nprocs -1:
#     Ts = all_Ts[rank*ops_per_rank:]
# else:
#     Ts = all_Ts[rank*ops_per_rank:(rank+1)*ops_per_rank]

Ts = [40, 430]
kB = 8.617e-5

# ******* 1: Load data *******
# arpackdir = path.expanduser('~/scratch/ArpackMAC/40x40')
# pos_dir = path.expanduser('~/scratch/clean_bigMAC/40x40/relax/relaxed_structures')
# sample_dir = f'sample-{sample_index}'

datadir = path.expanduser('~/Desktop/simulation_outputs/percolation/40x40')
edir = 'eARPACK'
Mdir = 'MOs_ARPACK'
posdir = 'structures'

mo_file = f'MOs_ARPACK_bigMAC-{sample_index}.npy'
energy_file = f'eARPACK_bigMAC-{sample_index}.npy'

# mo_path = path.join(arpackdir,sample_dir,mo_file)
# energy_path = path.join(arpackdir,sample_dir,energy_file)
# pos_path = path.join(pos_dir,sample_dir,'bigMAC-{sample_index}_relaxed.xsf')

mo_path = path.join(datadir,Mdir,mo_file)
energy_path = path.join(datadir,edir,energy_file)
pos_path = path.join(datadir,posdir,f'bigMAC-{sample_index}_relaxed.xsf')

energies = np.load(energy_path)
M =  np.load(mo_path)
pos, _ = read_xsf(pos_path)  


# ******* 2: Get gammas *******
# ga = 0.1 #edge atome-lead coupling in eV
# print("Computing AO gammas...")
# agaL, agaR = qcm.AO_gammas(pos,ga)
# print("Computing MO gammas...")
# gamL, gamR = qcm.MO_gammas(M,agaL, agaR, return_diag=True)
# np.save(f'gamL_40x40-{sample_index}.npy',gamL)
# np.save(f'gamR_40x40-{sample_index}.npy',gamR)

gamL = np.load(f'gamL_40x40-{sample_index}.npy')
gamR = np.load(f'gamR_40x40-{sample_index}.npy')


# ******* 3: Define strongly-coupled MOs *******
tolscal = 3.0
gamL_tol = np.mean(gamL) + tolscal*np.std(gamL)
gamR_tol = np.mean(gamR) + tolscal*np.std(gamR)

biggaL_inds = (gamL > gamL_tol).nonzero()[0]
biggaR_inds = (gamR > gamR_tol).nonzero()[0]

print(f'{biggaL_inds.shape[0]} MOs strongly coupled to left lead.')
print(f'{biggaR_inds.shape[0]} MOs strongly coupled to right lead.')


# ******* 4: Get a sense of the distance distribution *******
centres, ee, ii = generate_site_list(pos,M,energies)

centres = np.load('cc.npy')
ee = np.load('ee.npy')
ii = np.load('ii.npy')

print(ii)

cgamL = gamL[ii]
cgamR = gamR[ii]

for T in Ts:
    edArr, rdArr = diff_arrs(ee, centres, a0=30, eF=0)
    # ******* 5: Get spanning cluster *******
    conduction_clusters, dcrit, A = percolate(ee, pos, M, T, gamL_tol=gamL_tol,gamR_tol=gamR_tol, return_adjmat=True, distance='logMA',MOgams=(cgamL, cgamR), dArrs=(edArr,rdArr))

    with open(f'out_percolate-{T}K.pkl', 'wb') as fo:
        pickle.dump((conduction_clusters,dcrit,A), fo)

    c = conduction_clusters[0]
    plot_cluster(c, pos, M, A, usetex=True, show_densities=True, dotsize=1,centers=centres,inds=ii)
    print(conduction_clusters, dcrit)
