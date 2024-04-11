#!/usr/bin/env python

import sys
from os import path
import numpy as np
import qcnico.qchemMAC as qcm
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from percolate_mac.deploy_percolate import run_gridMOs

def load_data_dense(sample_index, size, compute_gammas=True):
    """ Loads atomic positions, energies, MOs, and coupling matrices of a given MAC structure.
    This function aims to be be common to all percolation runs (gridMOs or not, etc.). """


    arpackdir = path.expanduser(f'~/scratch/ArpackMAC/{size}')
    pos_dir = path.expanduser(f'~/scratch/clean_bigMAC/{size}/relax/relaxed_structures')

    posfile = f'bigMAC-{sample_index}_relaxed.xsf'


    mo_file = f'eigvecs-{sample_index}.npy'
    energy_file = f'eigvals-{sample_index}.npy'
    mo_path = path.join(arpackdir,'dense_tb_eigvecs',mo_file)
    energy_path = path.join(arpackdir,'dense_tb_eigvals',energy_file)

    energies = np.load(energy_path)
    M =  np.load(mo_path)
    
    pos_path = path.join(pos_dir,posfile)
    pos, _ = read_xsf(pos_path)

    rCC = 1.8
    pos = remove_dangling_carbons(pos, rCC)
    
    # ******* 2: Get gammas *******
    if compute_gammas:
        gamma = 0.1
        agaL, agaR = qcm.AO_gammas(pos, gamma)
        gamL, gamR = qcm.MO_gammas(M, agaL, agaR, return_diag=True)
        np.save(f'gamL-{sample_index}.npy', gamL)
        np.save(f'gamR-{sample_index}.npy', gamR)

    else:
        try:
            gamL = np.load(f'gamL-{sample_index}.npy')
            gamR = np.load(f'gamR-{sample_index}.npy')
        except FileNotFoundError:
            print('Gamma files not found. Re-computing gammas.')
            gamma = 0.1
            agaL, agaR = qcm.AO_gammas(pos, gamma)
            gamL, gamR = qcm.MO_gammas(M, agaL, agaR, return_diag=True)

            np.save(f'gamL-{sample_index}.npy', gamL)
            np.save(f'gamR-{sample_index}.npy', gamR)

    return pos, energies, M, gamL, gamR



n = int(sys.argv[1])
size = '20x20'

temps = np.arange(40,440,10)
dV = 0.0

pos, e, M, gamL, gamR = load_data_dense(n, size, compute_gammas=False)
run_gridMOs(pos,e,M,gamL,gamR,temps, dV, eF=0, compute_centres=False)
