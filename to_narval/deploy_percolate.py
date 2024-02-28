#!/usr/bin/env python

import sys
import pickle
from os import path
import numpy as np
import qcnico.qchemMAC as qcm
from qcnico.coords_io import read_xsf
from percolate import diff_arrs, percolate, generate_site_list

def load_data(sample_index, structype, motype,compute_gammas=True):

    """ Loads atomic positions, energies, MOs, and coupling matrices of a given MAC structure.
    This function aims to be be common to all percolation runs (gridMOs or not, etc.). """

    if structype == 'pCNN':

            arpackdir = path.expanduser('~/scratch/ArpackMAC/40x40')
            pos_dir = path.expanduser('~/scratch/clean_bigMAC/40x40/relax/no_PBC/relaxed_structures')

            posfile = f'bigMAC-{sample_index}_relaxed.xsf'

    else:

            arpackdir = path.expanduser(f'~/scratch/ArpackMAC/{structype}')
            pos_dir = path.expanduser(f'~/scratch/clean_bigMAC/{structype}/sample-{sample_index}/')

            posfile = f'{structype}n{sample_index}_relaxed.xsf'

    mo_file = f'MOs_ARPACK_bigMAC-{sample_index}.npy'
    energy_file = f'eARPACK_bigMAC-{sample_index}.npy'

    mo_path = path.join(arpackdir,'MOs',motype,mo_file)
    energy_path = path.join(arpackdir,'energies',motype,energy_file)
    pos_path = path.join(pos_dir,posfile)

    energies = np.load(energy_path)
    M =  np.load(mo_path)
    pos, _ = read_xsf(pos_path)  

    # ******* 2: Get gammas *******
    if compute_gammas:
        gamma = 0.1
        agaL, agaR = qcm.AO_gammas(pos, gamma)
        gamL, gamR = qcm.MO_gammas(M, agaL, agaR, return_diag=True)
        gamL = np.save(f'gamL_40x40-{sample_index}_{motype}.npy', gamL)
        gamR = np.save(f'gamR_40x40-{sample_index}_{motype}.npy', gamR)

    else:
        try:
            gamL = np.load(f'gamL_40x40-{sample_index}_{motype}.npy')
            gamR = np.load(f'gamR_40x40-{sample_index}_{motype}.npy')
        except FileNotFoundError:
            print('Gamma files not found. Re-computing gammas.')
            gamma = 0.1
            agaL, agaR = qcm.AO_gammas(pos, gamma)
            gamL, gamR = qcm.MO_gammas(M, agaL, agaR, return_diag=True)
            gamL = np.save(f'gamL_40x40-{sample_index}_{motype}.npy', gamL)
            gamR = np.save(f'gamR_40x40-{sample_index}_{motype}.npy', gamR)
    
    return pos, energies, M, gamL, gamR


def run_gridMOs(pos, energies, M,gamL, gamR, all_Ts, dV, tolscal=3.0, compute_centres=True, eF=0):
    # ******* Define strongly-coupled MOs *******
    gamL_tol = np.mean(gamL) + tolscal*np.std(gamL)
    gamR_tol = np.mean(gamR) + tolscal*np.std(gamR)

    L = set((gamL > gamL_tol).nonzero()[0])
    R = set((gamR > gamR_tol).nonzero()[0])

    # ******* Pre-compute distances *******

    if compute_centres:
        centres, ee, ii = generate_site_list(pos,M,L,R,energies,nbins=100)
        np.save(f'cc.npy',centres)
        np.save(f'ee.npy',ee)
        np.save(f'ii.npy', ii)    
    else:
        try:
            centres = np.load('cc.npy')
            ee = np.load('ee.npy')
            ii = np.load('ii.npy')
        except FileNotFoundError:
            print('Hopping centre files not found. Recomputing...')
            centres, ee, ii = generate_site_list(pos,M,L,R,energies,nbins=100)
            np.save(f'cc.npy',centres)
            np.save(f'ee.npy',ee)
            np.save(f'ii.npy', ii)
    
    if np.abs(dV) > 0:
        dX = np.max(pos[:,0]) - np.min(pos[:,0])
        E = np.array([dV/dX,0])
    else:
        E = np.array([0.0,0.0])

    edArr, rdArr = diff_arrs(ee, centres, a0=30, eF=eF, E=E)

    cgamL = gamL[ii]
    cgamR = gamR[ii]

    for T in all_Ts:
        # ******* 5: Get spanning cluster *******
        conduction_clusters, dcrit, A = percolate(ee, pos, M, T, gamL_tol=gamL_tol,gamR_tol=gamR_tol, return_adjmat=True, distance='logMA',MOgams=(cgamL, cgamR), dArrs=(edArr,rdArr))

        with open(f'out_percolate-{T}K.pkl', 'wb') as fo:
            pickle.dump((conduction_clusters,dcrit,A), fo)


def run_locMOs(pos, energies, M,gamL, gamR, all_Ts, eF, dV, tolscal=3.0):
    gamL_tol = np.mean(gamL) + tolscal*np.std(gamL)
    gamR_tol = np.mean(gamR) + tolscal*np.std(gamR)

    L = set((gamL > gamL_tol).nonzero()[0])
    R = set((gamR > gamR_tol).nonzero()[0])

    # ******* Pre-compute distances *******
    centres = qcm.MO_com(pos,M)

    if np.abs(dV) > 0:
        dX = np.max(centres[:,0]) - np.min(centres[:,0])
        E = np.array([dV/dX,0])
    else:
        E = np.array([0.0,0.0])
    
    a = np.mean(qcm.MO_rgyr(pos,M))
    
    edArr, rdArr = diff_arrs(energies, centres, a0=a, eF=eF, E=E)


    for T in all_Ts:
        # ******* 5: Get spanning cluster *******
        conduction_clusters, dcrit, A = percolate(energies, pos, M, T, gamL_tol=gamL_tol,gamR_tol=gamR_tol, return_adjmat=True, distance='logMA',MOgams=(gamL, gamR), dArrs=(edArr,rdArr), coupled_MO_sets=(L,R))

        with open(f'out_percolate-{T}K.pkl', 'wb') as fo:
            pickle.dump((conduction_clusters,dcrit,A), fo)