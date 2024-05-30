#!/usr/bin/env python

import pickle
from time import perf_counter
from os import path
import numpy as np
import qcnico.qchemMAC as qcm
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from .percolate import diff_arrs, percolate,\
        diff_arrs_w_inds, jitted_percolate, diff_arrs_var_a
from .MOs2sites import generate_site_list_opt

from .utils_arpackMAC import remove_redundant_eigenpairs

def load_data(sample_index, structype, motype='',compute_gammas=True,run_location='narval',save_gammas=False,gamma_dir='.',full_spectrum=False):
    """ Loads atomic positions, energies, MOs, and coupling matrices of a given MAC structure.
    This function aims to be be common to all percolation runs (gridMOs or not, etc.). """
    
    valid_run_locs = ['narval', 'local']
    assert run_location in valid_run_locs, f'Invalid value of argument run_location. Valid values:\n {valid_run_locs}'

    if run_location == 'narval':
        arpackdir = path.expanduser(f'~/scratch/ArpackMAC/{structype}')

        if structype == '40x40':
                pos_dir = path.expanduser('~/scratch/clean_bigMAC/40x40/relax/no_PBC/relaxed_structures')
                posfile = f'bigMAC-{sample_index}_relaxed.xsf'
        elif structype == '20x20':
                pos_dir = path.expanduser('~/scratch/clean_bigMAC/20x20/relax/relaxed_structures')
                posfile = f'bigMAC-{sample_index}_relaxed.xsf'

        else:
                arpackdir = path.expanduser(f'~/scratch/ArpackMAC/{structype}')
                pos_dir = path.expanduser(f'~/scratch/clean_bigMAC/{structype}/relaxed_structures/')
                posfile = f'{structype}n{sample_index}_relaxed.xsf'
        
        if full_spectrum:
            mo_dir = path.join(arpackdir,'dense_tb_eigvecs')
            e_dir = path.join(arpackdir,'dense_tb_eigvals')
        else:
            mo_dir = path.join(arpackdir,'MOs')
            e_dir = path.join(arpackdir,'energies')

    
    else: #running things locally
        percdir = path.expanduser('~/Desktop/simulation_outputs/percolation/')
        strucsize = '40x40'
        if structype == 'pCNN':
            mo_dir = path.join(percdir, strucsize,'MOs_ARPACK')
            e_dir = path.join(percdir, strucsize, 'eARPACK')
            pos_dir = path.join(percdir, strucsize, 'structures')
            posfile = f'bigMAC-{sample_index}_relaxed.xsf'
        else:
            print('not implemented. returning 0. if running locally, structype must be "pCNN".')
            return 0

    if full_spectrum:
        mo_file = f'eigvecs-{sample_index}.npy'
        energy_file = f'eigvals-{sample_index}.npy'
    else:
        mo_file = f'MOs_ARPACK_bigMAC-{sample_index}.npy'
        energy_file = f'eARPACK_bigMAC-{sample_index}.npy'
    mo_path = path.join(mo_dir,motype,mo_file)
    energy_path = path.join(e_dir,motype,energy_file)

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
        if save_gammas:
            np.save(path.join(gamma_dir, f'gamL_40x40-{sample_index}_{motype}.npy', gamL))
            np.save(path.join(gamma_dir, f'gamR_40x40-{sample_index}_{motype}.npy', gamR))

    else:
        try:
            gamL = np.load(path.join(gamma_dir, f'gamL_40x40-{sample_index}_{motype}.npy'))
            gamR = np.load(path.join(gamma_dir, f'gamR_40x40-{sample_index}_{motype}.npy'))
        except FileNotFoundError:
            print('Gamma files not found. Re-computing gammas.')
            gamma = 0.1
            agaL, agaR = qcm.AO_gammas(pos, gamma)
            gamL, gamR = qcm.MO_gammas(M, agaL, agaR, return_diag=True)
            np.save(path.join(gamma_dir, f'gamL_40x40-{sample_index}_{motype}.npy'), gamL)
            np.save(path.join(gamma_dir, f'gamR_40x40-{sample_index}_{motype}.npy'), gamR)
    
    return pos, energies, M, gamL, gamR


def load_var_a_data(datadir='.'):
    sites_pos = np.load(path.join(datadir, 'var_a_npys/centers.npy'))
    sites_energies = np.load(path.join(datadir, 'var_a_npys/ee.npy'))
    sites_radii = np.load(path.join(datadir, 'var_a_npys/radii.npy'))
    ii = np.load(path.join(datadir, 'var_a_npys/ii.npy'))

    return sites_pos, sites_energies, sites_radii, ii



def load_data_multi(sample_index, structype, motypes, e_file_names=None, MO_file_names=None,compute_gammas=True):
    """Same as `load_data`, but this time loads data from multiple diagonalisation runs to 
    use more eigenpairs for percolation calculation."""

    if structype == 'pCNN':
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


def setup_hopping_sites_gridMOs(pos, energies, M, gamL, gamR, tolscal=3.0, nbins=100, compute_centres=True, datapath='.', save_centers=True, return_ii=False):
    """Once all of the data relevant to a MAC structure has been loaded (atomic positions, MOs, energies, lead-coupled MOs), this function obtains the hopping sites,
    either by calling `generate_site_list`, or by reading NPY files in the folder specified by `datapath`."""
    # ******* Define strongly-coupled MOs *******
    gamL_tol = np.mean(gamL) + tolscal*np.std(gamL)
    gamR_tol = np.mean(gamR) + tolscal*np.std(gamR)

    # Set of MOs (not sites!) which are strongly coupled to the leads 
    L_mos = set((gamL > gamL_tol).nonzero()[0])
    R_mos = set((gamR > gamR_tol).nonzero()[0])

    # ******* Pre-compute distances *******
    if compute_centres:
        
        centres, ee, ii = generate_site_list_opt(pos,M,L_mos,R_mos,energies,nbins=100)
        if save_centers:
            np.save(path.join(datapath,f'cc.npy'),centres)
            np.save(path.join(datapath,f'ee.npy'),ee)
            np.save(path.join(datapath,f'ii.npy'), ii)
    else:
        try:
            centres = np.load(path.join(datapath, 'cc.npy'))
            ee = np.load(path.join(datapath, 'ee.npy'))
            ii = np.load(path.join(datapath, 'ii.npy'))
        except FileNotFoundError:
            print('Hopping centre files not found. Recomputing...')
            centres, ee, ii = generate_site_list_opt(pos,M,L_mos,R_mos,energies,nbins=nbins)
            np.save(path.join(datapath, f'cc.npy',centres))
            np.save(path.join(datapath, f'ee.npy',ee))
            np.save(path.join(datapath, f'ii.npy', ii))
    
    cgamL = gamL[ii]
    cgamR = gamR[ii]
    
    L_sites = set((cgamL > gamL_tol).nonzero()[0])
    R_sites = set((cgamR > gamR_tol).nonzero()[0])
        
    if return_ii: # ii[n] is the index of the MO which yielded the site at centres[n] (w energy ee[n])
        return centres, ee, L_sites, R_sites, ii
    else:
        return centres, ee, L_sites, R_sites

def run_percolate(sites_pos, sites_energies, L, R, all_Ts, dV, eF=0, a0=30, pkl_dir='.',jitted=False,pkl_prefix='out_percolate',rmax=None):
    """Once the hopping sites have been defined, run `percolate` under the array of desired conditions (temperature, external field).
    This function returns nothing; it writes the results of each call to `percolate` to a pickle file located in `pkl_dir`."""
    kB = 8.617333262 #eV / K
    if np.abs(dV) > 0:
        dX = np.max(sites_pos[:,0]) - np.min(sites_pos[:,0])
        E = np.array([dV/dX,0])
    else:
        E = np.array([0.0,0.0])
    
    if isinstance(a0, np.ndarray):
        print('~~~~ a0 is an array: running variable radii percolation ~~~~')
        if rmax is not None:
            igood = (a0 < rmax).nonzero()[0] # filter out sites that are 'too big'
            sites_pos = sites_pos[igood]
            sites_energies = sites_energies[igood]
            a0 = a0[igood]
            
        edArr, rdArr, ij = diff_arrs_var_a(sites_energies, sites_pos, radii=a0, eF=eF, E=E)
        var_a = True
    else:
        print('~~~~ a0 is a scalar: running variable single-radius percolation ~~~~')
        edArr, rdArr, ij = diff_arrs_w_inds(sites_energies, sites_pos, a0=a0, eF=eF, E=E)
        var_a = False
    
    k=0

    #all_Ts = np.sort(all_Ts)[::-1] # sort in reverse order to leverage `prev_d_ind` speedup
    prev_d_ind = 0
    for T in all_Ts:
        print(f'******* T = {T} K *******')
        darr = rdArr + (edArr / (kB * T))
        start = perf_counter()
        if jitted:
            conduction_clusters, dcrit, A = jitted_percolate(darr,ij,L,R)
        else:
            conduction_clusters, dcrit, A, prev_d_ind = percolate(darr,ij,L,R,return_adjmat=True,prev_d_ind=0)
        end = perf_counter()
        if k >0: # don't time first call to percolate (avoid measuring compilation time)
            print(f'\n~~~Done! [{end-start} seconds]~~~\n Saving to pkl file...~~~',flush=True)


        
        pkl_name = pkl_prefix +  f'-{T}K.pkl'

        with open(path.join(pkl_dir, pkl_name), 'wb') as fo:
            pickle.dump((conduction_clusters,dcrit,A), fo)
        print('Saved successfully.\n', flush=True)
        k+=1


def run_percolate_locMOs(pos, energies, M,gamL, gamR, all_Ts, eF, dV, tolscal=3.0, pkl_dir='.'):
    """This function combines `setup_hopping_sites` and `run_percolate` for the cases where we want to 
    do percolation on sites obtained without gridifying each MO."""
    kB = 8.617333262 #eV / K
    gamL_tol = np.mean(gamL) + tolscal*np.std(gamL)
    gamR_tol = np.mean(gamR) + tolscal*np.std(gamR)

    L = set((gamL > gamL_tol).nonzero()[0])
    R = set((gamR > gamR_tol).nonzero()[0])

    centres = qcm.MO_com(pos,M)

    if np.abs(dV) > 0:
        dX = np.max(centres[:,0]) - np.min(centres[:,0])
        E = np.array([dV/dX,0])
    else:
        E = np.array([0.0,0.0])
    
    a = np.mean(qcm.MO_rgyr(pos,M))
    
    edArr, rdArr = diff_arrs(energies, centres, a0=a, eF=eF, E=E)
    for T in all_Ts:
        print(f'******* T = {T} K *******')
        darr = rdArr + (edArr / (kB * T))
        conduction_clusters, dcrit, A = percolate(darr,L,R,return_adjmat=True)
        print('\nDone! Saving to pkl file...')

        with open(path.join(pkl_dir, f'out_percolate-{T}K.pkl'), 'wb') as fo:
            pickle.dump((conduction_clusters,dcrit,A), fo)
        print('Saved successfully.\n')
