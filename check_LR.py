#!/usr/bin/env python

import numpy as np
from qcnico.qchemMAC import AO_gammas, MO_gammas
from qcnico.coords_io import read_xyz
from qcnico.data_utils import save_npy
import os
import sys
from time import perf_counter


def compute_gammas(pos, M, gamma=0.1):
    print("Getting AO gammas...", end=' ', flush=True)
    start = perf_counter()
    agaL, agaR = AO_gammas(pos,gamma)
    ao_end = perf_counter()
    print(f"Done! [{ao_end - start} seconds].\nGetting MO gammas...", end=' ', flush=True)
    gamL, gamR = MO_gammas(M,agaL,agaR, return_diag=True)
    mo_end=perf_counter()
    print(f"Done! [{mo_end - ao_end} seconds].")
    
    return gamL, gamR


def filter_LR(gamL, gamR,tolscal=3.0):
    gamL_tol = np.mean(gamL) + tolscal * np.std(gamL)
    gamR_tol = np.mean(gamR) + tolscal * np.std(gamR)

    L = set((gamL > gamL_tol).nonzero()[0])
    R = set((gamR > gamR_tol).nonzero()[0])

    return L, R



nn = int(sys.argv[1])
runtype = 'MOs_pure'
motype = sys.argv[2]


if runtype not in ['MOs_pure', 'sites_pure', 'mixed']:
    print(f'Invalid `runtype` {runtype}. Valid entries are:\n* "mixed": sites created by k-clustering, MO gammas;\n*"MOs_pure": MOs directly as sites;\n*"sites_pure": sites created by k-clustering, gammas obtained from site kets.\nExiting angrily.')
    sys.exit()

subdir = runtype

structype = os.path.basename(os.getcwd())
strucdir = os.path.expanduser(f"~/scratch/clean_bigMAC/{structype}/relaxed_structures_no_dangle/")

if structype == '40x40':
    rmax = 18.03
    strucfile = os.path.join(strucdir, f'bigMAC-{nn}_relaxed_no-dangle.xyz')
elif structype == 'tempdot6':
    rmax=121.2
    strucfile = os.path.join(strucdir, f'{structype}n{nn}_relaxed_no-dangle.xyz')
elif structype == 'tempdot5':
    rmax = 198.69
    strucfile = os.path.join(strucdir, f'{structype}n{nn}_relaxed_no-dangle.xyz')
else:
    print(f'Structure type {structype} is invalid. Exiting angrily.')
    sys.exit()

pos = read_xyz(strucfile)

if runtype == 'sites_pure' or runtype == 'mixed':

    gammas_dir = f'sample-{nn}/gammas/{runtype}_rmax_{rmax}'

    if motype == 'virtual':
        sitesdir = f'sample-{nn}/sites_data_0.00105_psi_pow2/'
    else:
        sitesdir = f'sample-{nn}/sites_data_0.00105_psi_pow2_{motype}/'

    radii = np.load(os.path.join(sitesdir, f'radii.npy'))   
    rfilter = radii < rmax

if runtype == 'sites_pure':
    M = np.load(os.path.join(sitesdir, f'site_state_matrix.npy'))
    M = M[:,rfilter]
    M /= np.linalg.norm(M,axis=0)
    nsites = M.shape[1]
    
    
    if not os.path.exists(os.path.join(gammas_dir, f'gamL_{motype}')): # if gamL file does not exist; assume gamR doesn't either; compute both
        gamL, gamR = compute_gammas(pos, M)
        save_npy(gamL, f'gamL_{motype}', gammas_dir)
        save_npy(gamR, f'gamR_{motype}', gammas_dir)
    L, R = filter_LR(gamL, gamR)


else: #kets_type == 'MOs':

    gammas_dir = f'sample-{nn}/gammas/{runtype}' 

    if motype == 'virtual':
        M = np.load(os.path.expanduser(f"~/scratch/ArpackMAC/{structype}/MOs/{motype}/MOs_ARPACK_bigMAC-{nn}.npy"))
    else:
        M = np.load(os.path.expanduser(f"~/scratch/ArpackMAC/{structype}/MOs/{motype}/MOs_ARPACK_{motype}_{structype}-{nn}.npy"))

    if not os.path.exists(os.path.join(gammas_dir, f'gamL_{motype}')): # if gamL file does not exist; assume gamR doesn't either; compute both
        gamL, gamR = compute_gammas(pos, M)
        save_npy(gamL, f'gamL_{motype}', gammas_dir)
        save_npy(gamR, f'gamR_{motype}', gammas_dir)
    L, R = filter_LR(gamL, gamR)

    if runtype == 'mixed':
        ii = np.load(os.path.join(sitesdir, 'ii.npy'))
        ii = set(ii[rfilter]) # indices of MOs yielding sites actually used in the percolation run

        nsites = ii.shape[0]

        L = L & ii # left-coupled MOs used in percolation run
        R = R & ii # right-coupled MOs used in percolation run
    else:
        nsites = M.shape[1]
     

print("Total # of sites = ", nsites)
print("# of L sites = ", len(L))
print("# of R sites = ", len(R))