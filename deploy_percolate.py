#!/usr/bin/env python

import sys
import pickle
from time import perf_counter
from os import path
import numpy as np
import qcnico.qchemMAC as qcm
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from .percolate import diff_arrs, percolate, generate_site_list, diff_arrs_var_a
from utils_arpackMAC import remove_redundant_eigenpairs

def load_data(sample_index, structype, motype,gammas_method='compute'):
    """ Loads atomic positions, energies, MOs, and coupling matrices of a given MAC structure.
    This function aims to be be common to all percolation runs (gridMOs or not, etc.). """

    if structype == '40x40':
        pos_dir = path.expanduser('~/scratch/clean_bigMAC/40x40/relaxed_structures_no_dangle/')
        posfile = f'bigMAC-{sample_index}_relaxed_no-dangle.xyz'

    else:
        pos_dir = path.expanduser(f'~/scratch/clean_bigMAC/{structype}/relaxed_structures_no_dangle/')
        posfile = f'{structype}n{sample_index}_relaxed.xsf'
    
    arpackdir = path.expanduser(f'~/scratch/ArpackMAC/{structype}')

    mo_file = f'MOs_ARPACK_bigMAC-{sample_index}.npy'
    energy_file = f'eARPACK_bigMAC-{sample_index}.npy'
    mo_path = path.join(arpackdir,'MOs',motype,mo_file)
    energy_path = path.join(arpackdir,'energies',motype,energy_file)

    energies = np.load(energy_path)
    M =  np.load(mo_path)
    
    pos_path = path.join(pos_dir,posfile)
    pos, _ = read_xsf(pos_path)

    rCC = 1.8
    pos = remove_dangling_carbons(pos, rCC)
    
    # ******* 2: Get gammas *******
    if gammas_method == 'compute':
        gamma = 0.1
        agaL, agaR = qcm.AO_gammas(pos, gamma)
        gamL, gamR = qcm.MO_gammas(M, agaL, agaR, return_diag=True)
        np.save(f'gamL_40x40-{sample_index}_{motype}.npy', gamL)
        np.save(f'gamR_40x40-{sample_index}_{motype}.npy', gamR)
        
        return pos, energies, M, gamL, gamR

    elif gammas_method == 'load':
        try:
            gamL = np.load(f'gamL_40x40-{sample_index}_{motype}.npy')
            gamR = np.load(f'gamR_40x40-{sample_index}_{motype}.npy')
        except FileNotFoundError:
            print('Gamma files not found. Re-computing gammas.')
            gamma = 0.1
            agaL, agaR = qcm.AO_gammas(pos, gamma)
            gamL, gamR = qcm.MO_gammas(M, agaL, agaR, return_diag=True)
            np.save(f'gamL_40x40-{sample_index}_{motype}.npy', gamL)
            np.save(f'gamR_40x40-{sample_index}_{motype}.npy', gamR)
    
        return pos, energies, M, gamL, gamR
    elif gammas_method == 'none':
        return pos, energies, M
    else:
        print(f'''[load_data ERROR] Invalid for entry for `gammas_method`: {gammas_method}.
              \nValid entries are:\n
              * "compute" (default): Computes all of the MO-lead couplings from scratch\n
              * "load": Checks if MO-lead couplings have already been computed in saved in current directory. If yes, load them from NPY files. If no, computes them from scratch (same as "compute")\n
              * "none": does not retrieve the couplings; returns only pos, energies, and MO matrix.\n
            Assuming "none" here.
              ''')
        return pos, energies, M



def load_data_multi(sample_index, structype, motypes, e_file_names=None, MO_file_names=None,compute_gammas=True):
    """Same as `load_data`, but this time loads data from multiple diagonalisation runs to 
    use more eigenpairs for percolation calculation."""

    if structype == 'pCNN' or '40x40':
            arpackdir = path.expanduser('~/scratch/ArpackMAC/40x40')
            pos_dir = path.expanduser('~/scratch/clean_bigMAC/40x40/relax/no_PBC/relaxed_structures')

            posfile = f'bigMAC-{sample_index}_relaxed.xsf'
    else:
            arpackdir = path.expanduser(f'~/scratch/ArpackMAC/{structype}')
            pos_dir = path.expanduser(f'~/scratch/clean_bigMAC/{structype}/sample-{sample_index}/')

            posfile = f'{structype}n{sample_index}_relaxed.xsf'

    if MO_file_names is None:
        mo_files = [f'MOs_ARPACK_bigMAC-{sample_index}.npy'] * len(motypes)
    else:
        mo_files = [mfn + f'-{sample_index}.npy' for mfn in MO_file_names]

    if e_file_names is None:
        energy_files = [f'eARPACK_bigMAC-{sample_index}.npy'] * len(motypes)
    else:
        energy_files = [efn + f'-{sample_index}.npy' for efn in e_file_names]


    mo_paths = [path.join(arpackdir,'MOs',motype,mo_file) for (motype, mo_file) in zip(motypes, mo_files)]
    energy_paths = [path.join(arpackdir,'energies',motype, energy_file) for (motype, energy_file) in zip(motypes, energy_files)]

    energies = np.hstack([np.load(energy_path) for energy_path in energy_paths])
    M =  np.hstack([np.load(mo_path) for mo_path in mo_paths])

    energies, M = remove_redundant_eigenpairs(energies, M)
    
    pos_path = path.join(pos_dir,posfile)
    pos, _ = read_xsf(pos_path)  
    
    rCC = 1.8
    pos = remove_dangling_carbons(pos, rCC)

    # ******* 2: Get gammas *******
    if compute_gammas:
        gamma = 0.1
        agaL, agaR = qcm.AO_gammas(pos, gamma)
        gamL, gamR = qcm.MO_gammas(M, agaL, agaR, return_diag=True)
        np.save(f'gamL_40x40-{sample_index}.npy', gamL)
        np.save(f'gamR_40x40-{sample_index}.npy', gamR)

    else:
        try:
            gamL = np.load(f'gamL_40x40-{sample_index}.npy')
            gamR = np.load(f'gamR_40x40-{sample_index}.npy')
        except FileNotFoundError:
            print('Gamma files not found. Re-computing gammas.')
            gamma = 0.1
            agaL, agaR = qcm.AO_gammas(pos, gamma)
            gamL, gamR = qcm.MO_gammas(M, agaL, agaR, return_diag=True)
            np.save(f'gamL_40x40-{sample_index}.npy', gamL)
            np.save(f'gamR_40x40-{sample_index}.npy', gamR)
    
    return pos, energies, M, gamL, gamR


def run_gridMOs(pos, energies, M,gamL, gamR, all_Ts, dV, tolscal=3.0, compute_centres=True, eF=0, nbins=100):
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
            centres, ee, ii = generate_site_list(pos,M,L,R,energies,nbins=nbins)
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

    all_Ts = np.flip(np.sort(all_Ts))
    d_prev_ind = 0
    for T in all_Ts:
        # ******* 5: Get spanning cluster *******
        conduction_clusters, dcrit, A, iidprev = percolate(ee, pos, M, T, gamL_tol=gamL_tol,gamR_tol=gamR_tol, return_adjmat=True, distance='logMA',MOgams=(cgamL, cgamR), dArrs=(edArr,rdArr),prev_d_ind=d_prev_ind)
        d_prev_ind = iidprev
        print(f'Distance nb {d_prev_ind} yielded a percolating cluster!', flush=True)

        with open(f'out_percolate-{T}K.pkl', 'wb') as fo:
            pickle.dump((conduction_clusters,dcrit,A), fo)

def run_var_a(pos, M, gamL, gamR, all_Ts, dV, tolscal=3.0, eF=0, hyperlocal=False,npydir='./var_a_npys',run_name=None,rmax=None,rho_min=None,use_idprev=True,check_sites=False):
    # ******* Define strongly-coupled MOs *******
    gamL_tol = np.mean(gamL) + tolscal*np.std(gamL)
    gamR_tol = np.mean(gamR) + tolscal*np.std(gamR)

    # ******* Pre-compute distances *******
    if hyperlocal:
        centres = np.load(path.join(npydir,'centers_hl.npy'))
        ee = np.load(path.join(npydir,'ee_hl.npy'))
        ii = np.load(path.join(npydir, 'ii_hl.npy'))
        radii = np.load(path.join(npydir, 'radii_hl.npy'))
        dat = np.load(path.join(npydir, 'rr_v_masses_v_iprs_v_ee_hl.npy'))
        radii, masses, _, ee = dat
    else:
        centres = np.load(path.join(npydir,'centers.npy'))
        ee = np.load(path.join(npydir,'ee.npy'))
        ii = np.load(path.join(npydir, 'ii.npy'))
        radii = np.load(path.join(npydir, 'radii.npy'))
        dat = np.load(path.join(npydir, 'rr_v_masses_v_iprs_v_ee.npy'))
        radii, masses, _, ee = dat
    
    if check_sites: #making sure that the pre-computed sites/radii correspond to the site kets stored in site_state_matrix
        S = np.load(path.join(npydir, 'site_state_matrix.npy'))
        nradii = radii.shape[0]
        nsites = centres.shape[0]
        print('centers.shape[0] == radii.shape[0]: ', nsites == nradii, flush=True)
        print('centers.shape[0] == S.shape[1] (nb. of cols in site ket matrix): ', nsites == S.shape[1], flush=True)
        recomputed_centres = np.zeros_like(centres)
        recomputed_radii = np.zeros_like(radii)
        for n in range(nsites):
            # here we assume the `hyperlocal` kwarh in `generate_site_radii_list` is set to `sites` (default)
            recomputed_centres[n,:] = qcm.MO_com_hyperlocal(pos, S, n)
            recomputed_radii = qcm.MO_rgyr(pos, S, n, renormalise=True)
        
        rdiff = np.abs(radii - recomputed_radii)
        print(f'**** MAX DIFFERENCE BETWEEN RADII = {np.max(rdiff)} ****')
        print(f'**** MAX DISTANCE BETWEEN CENTRES = {np.max(np.linalg.norm(centres - recomputed_centres,axis=1))} ****')
            
            
    if np.abs(dV) > 0:
        dX = np.max(pos[:,0]) - np.min(pos[:,0])
        E = np.array([dV/dX,0])
    else:
        E = np.array([0.0,0.0])

    if rmax is not None:
        igood = (radii < rmax).nonzero()[0]
        centres = centres[igood]
        ee = ee[igood]
        ii = ii[igood]
        radii = radii[igood]

    if rho_min is not None:
        rhos = masses / (np.pi * radii * radii)
        igood = (rhos > rho_min) 
        centres = centres[igood]
        ee = ee[igood]
        ii = ii[igood]
        radii = radii[igood]


    edArr, rdArr = diff_arrs_var_a(ee, centres, radii, eF=eF, E=E)

    cgamL = gamL[ii]
    cgamR = gamR[ii]

    all_Ts = np.flip(np.sort(all_Ts))
    d_prev_ind = 0
    tracker_file = f'finished_temps_{run_name}.txt'
    ftrack = open(tracker_file,'w')

    for T in all_Ts:
        # ******* 5: Get spanning cluster *******
        conduction_clusters, dcrit, A, iidprev = percolate(ee, pos, M, T, gamL_tol=gamL_tol,gamR_tol=gamR_tol, return_adjmat=True, distance='logMA',MOgams=(cgamL, cgamR), dArrs=(edArr,rdArr),prev_d_ind=d_prev_ind)
        if use_idprev:
            d_prev_ind = iidprev #update minimum index of distances to consider, if use_idprev, otherwise first ind to look at is always 0
        print(f'Distance nb {d_prev_ind} yielded a percolating cluster!', flush=True)
        
        if run_name is None:
            if hyperlocal:
                pkl_name = f'out_var_a_hl_rollback_percolate-{T}K.pkl'
            else:
                pkl_name = f'out_var_a_rollback_percolate-{T}K.pkl'
        else:
            pkl_name = 'out_percolate_' + run_name + f'-{T}K.pkl'

        with open(pkl_name, 'wb') as fo:
            pickle.dump((conduction_clusters,dcrit,A), fo)
        ftrack.write(f'{T}K\n')
    ftrack.close()


def run_var_a_from_sites(pos, M, S, all_Ts, dV, tol_scal=3.0 ,eF=0, hyperlocal=False,npydir='./var_a_npys',run_name=None,rmax=None, sGammas=None,gamma=0.1):
    
    # Ensure site kets are properly normalised
    S /= np.linalg.norm(S,axis=0)



    # ******* Pre-compute distances *******
    if hyperlocal:
        centres = np.load(path.join(npydir,'centers_hl.npy'))
        ee = np.load(path.join(npydir,'ee_hl.npy'))
        radii = np.load(path.join(npydir, 'radii_hl.npy'))

    else:
        centres = np.load(path.join(npydir,'centers.npy'))
        ee = np.load(path.join(npydir,'ee.npy'))
        radii = np.load(path.join(npydir, 'radii.npy'))
    
    if np.abs(dV) > 0:
        dX = np.max(pos[:,0]) - np.min(pos[:,0])
        E = np.array([dV/dX,0])
    else:
        E = np.array([0.0,0.0])

    if rmax is not None:
        igood = (radii < rmax).nonzero()[0]
        centres = centres[igood]
        ee = ee[igood]
        radii = radii[igood]
        S = S[:,igood]

    if sGammas is None:
        print('Getting AO gamma...', end=' ')
        start = perf_counter()
        agaL, agaR = qcm.AO_gammas(pos, gamma)
        ao_end = perf_counter()
        print(f'Done! [{ao_end - start} seconds]\nGetting sites gammas...', end=' ')
        cgamL, cgamR = qcm.MO_gammas(S, agaL, agaR, return_diag=True)
        end = perf_counter()
        print(f'Done! [{end - ao_end} seconds]')
    else:
        cgamL, cgamR = sGammas

    gamL_tol = np.mean(cgamL) + tol_scal * np.std(cgamL)
    gamR_tol = np.mean(cgamR) + tol_scal * np.std(cgamR)
    
    edArr, rdArr = diff_arrs_var_a(ee, centres, radii, eF=eF, E=E)
    
    tracker_file = f'finished_temps_{run_name}_sites_var_a.txt'
    ftrack = open(tracker_file,'w')

    for T in all_Ts:
        # ******* 5: Get spanning cluster *******
        conduction_clusters, dcrit, A, iidprev = percolate(ee, pos, M, T, gamL_tol=gamL_tol,gamR_tol=gamR_tol, return_adjmat=True, distance='logMA',MOgams=(cgamL, cgamR), dArrs=(edArr,rdArr))
        
        if run_name is None:
            if hyperlocal:
                pkl_name = f'out_var_a_hl_rollback_percolate-{T}K.pkl'
            else:
                pkl_name = f'out_var_a_rollback_percolate-{T}K.pkl'
        else:
            pkl_name = 'out_percolate_' + run_name + f'-{T}K.pkl'

        with open(pkl_name, 'wb') as fo:
            pickle.dump((conduction_clusters,dcrit,A), fo)
        ftrack.write(f'{T}K\n')
    ftrack.close()


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
