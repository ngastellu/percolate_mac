#!/usr/bin/env python

import pickle
from time import perf_counter
import os
import numpy as np
import qcnico.qchemMAC as qcm
from qcnico.coords_io import read_xsf, read_xyz
from qcnico.rotate_pos import rotate_pos
from qcnico.data_utils import save_npy
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from .percolate import diff_arrs, percolate,\
        diff_arrs_w_inds, jitted_percolate, diff_arrs_var_a
from .MOs2sites import generate_site_list_opt
from .utils_arpackMAC import remove_redundant_eigenpairs


def load_data(sample_index, structype, motype,gammas_method='compute',sort_energies=False):
    """ Loads atomic positions, energies, MOs, and coupling matrices of a given MAC structure.
    This function aims to be be common to all percolation runs (gridMOs or not, etc.). """

    if structype == '40x40':
        pos_dir = os.path.expanduser('~/scratch/clean_bigMAC/40x40/relaxed_structures_no_dangle/')
        posfile = f'bigMAC-{sample_index}_relaxed_no-dangle.xyz'

    else:
        pos_dir = os.path.expanduser(f'~/scratch/clean_bigMAC/{structype}/relaxed_structures_no_dangle/')
        posfile = f'{structype}n{sample_index}_relaxed_no-dangle.xyz'
    
    arpackdir = os.path.expanduser(f'~/scratch/ArpackMAC/{structype}')

    
    if motype == 'virtual' or motype == 'occupied' or motype == 'virtual_w_HOMO':
        mo_file = f'MOs_ARPACK_bigMAC-{sample_index}.npy'
        energy_file = f'eARPACK_bigMAC-{sample_index}.npy'
    
    else: #'hi' or 'lo'
        mo_file = f'MOs_ARPACK_{motype}_{structype}-{sample_index}.npy'
        energy_file = f'eARPACK_{motype}_{structype}-{sample_index}.npy'
    
    mo_path = os.path.join(arpackdir,'MOs',motype,mo_file)
    energy_path = os.path.join(arpackdir,'energies',motype,energy_file)

    energies = np.load(energy_path)
    M =  np.load(mo_path)

    # Sort energies/MOs
    if sort_energies:
        eii = np.argsort(energies)
        energies = energies[eii]
        M = M[:,eii]
    
    pos_path = os.path.join(pos_dir,posfile)
    pos = read_xyz(pos_path)
 
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



def load_var_a_data(datadir='.'):
    sites_pos = np.load(os.path.join(datadir, 'centers.npy'))
    sites_energies = np.load(os.path.join(datadir, 'ee.npy'))
    sites_radii = np.load(os.path.join(datadir, 'radii.npy'))
    ii = np.load(os.path.join(datadir, 'ii.npy'))

    return sites_pos, sites_energies, sites_radii, ii



def load_data_multi(sample_index, structype, motypes, e_file_names=None, MO_file_names=None,compute_gammas=True):
    """Same as `load_data`, but this time loads data from multiple diagonalisation runs to 
    use more eigenpairs for percolation calculation."""

    if structype == 'pCNN':
            arpackdir = os.path.expanduser('~/scratch/ArpackMAC/40x40')
            pos_dir = os.path.expanduser('~/scratch/clean_bigMAC/40x40/relax/no_PBC/relaxed_structures')

            posfile = f'bigMAC-{sample_index}_relaxed.xsf'
    else:
            arpackdir = os.path.expanduser(f'~/scratch/ArpackMAC/{structype}')
            pos_dir = os.path.expanduser(f'~/scratch/clean_bigMAC/{structype}/sample-{sample_index}/')

            posfile = f'{structype}n{sample_index}_relaxed.xsf'

    if MO_file_names is None:
        mo_files = [f'MOs_ARPACK_bigMAC-{sample_index}.npy'] * len(motypes)
    else:
        mo_files = [mfn + f'-{sample_index}.npy' for mfn in MO_file_names]

    if e_file_names is None:
        energy_files = [f'eARPACK_bigMAC-{sample_index}.npy'] * len(motypes)
    else:
        energy_files = [efn + f'-{sample_index}.npy' for efn in e_file_names]


    mo_paths = [os.path.join(arpackdir,'MOs',motype,mo_file) for (motype, mo_file) in zip(motypes, mo_files)]
    energy_paths = [os.path.join(arpackdir,'energies',motype, energy_file) for (motype, energy_file) in zip(motypes, energy_files)]

    energies = np.hstack([np.load(energy_path) for energy_path in energy_paths])
    M =  np.hstack([np.load(mo_path) for mo_path in mo_paths])

    energies, M = remove_redundant_eigenpairs(energies, M)
    
    pos_path = os.path.join(pos_dir,posfile)
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
            np.save(os.path.join(datapath,f'cc.npy'),centres)
            np.save(os.path.join(datapath,f'ee.npy'),ee)
            np.save(os.path.join(datapath,f'ii.npy'), ii)
    else:
        try:
            centres = np.load(os.path.join(datapath, 'cc.npy'))
            ee = np.load(os.path.join(datapath, 'ee.npy'))
            ii = np.load(os.path.join(datapath, 'ii.npy'))
        except FileNotFoundError:
            print('Hopping centre files not found. Recomputing...')
            centres, ee, ii = generate_site_list_opt(pos,M,L_mos,R_mos,energies,nbins=nbins)
            np.save(os.path.join(datapath, f'cc.npy',centres))
            np.save(os.path.join(datapath, f'ee.npy',ee))
            np.save(os.path.join(datapath, f'ii.npy', ii))
    
    cgamL = gamL[ii]
    cgamR = gamR[ii]
    
    L_sites = set((cgamL > gamL_tol).nonzero()[0])
    R_sites = set((cgamR > gamR_tol).nonzero()[0])
        
    if return_ii: # ii[n] is the index of the MO which yielded the site at centres[n] (w energy ee[n])
        return centres, ee, L_sites, R_sites, ii
    else:
        return centres, ee, L_sites, R_sites


def run_percolate_from_sites(pos, M, S, all_Ts, dV, tol_scal=3.0 ,eF=0, hyperlocal=False,npydir='./var_a_npys',run_name=None,rmax=None, dE_max=None, sGammas=None,gamma=0.1,pkl_dir='.',rotate=False,dcrits_npy=True):
    
    if not os.path.isdir(pkl_dir):
        os.mkdir(pkl_dir)

    # Ensure site kets are properly normalised
    S /= np.linalg.norm(S,axis=0)



    # ******* Pre-compute distances *******
    if hyperlocal:
        centres = np.load(os.path.join(npydir,'centers_hl.npy'))
        ee = np.load(os.path.join(npydir,'ee_hl.npy'))
        radii = np.load(os.path.join(npydir, 'radii_hl.npy'))

    else:
        centres = np.load(os.path.join(npydir,'centers.npy'))
        ee = np.load(os.path.join(npydir,'ee.npy'))
        radii = np.load(os.path.join(npydir, 'radii.npy'))
    
    if np.abs(dV) > 0:
        dX = np.max(pos[:,0]) - np.min(pos[:,0])
        E = np.array([dV/dX,0])
    else:
        E = np.array([0.0,0.0])
    
    nsites = ee.shape[0]

    if rmax is not None:
        rfilter = radii < rmax
    else:
        rfilter = np.ones(nsites,dtype='bool')

    if dE_max is not None:
        efilter = np.abs(ee - eF) <= dE_max # max allowed distance from Fermi energy
    else:
        efilter = np.ones(nsites, dtype='bool')
    
    igood = (rfilter * efilter).nonzero()[0] # apply both filters, get remaining valid indices

    centres = centres[igood]
    ee = ee[igood]
    radii = radii[igood]
    S = S[:,igood]

    np.save(os.path.join(pkl_dir, 'igood.npy'), igood)
    print('saved igood!', flush=True)
    
    if rotate:
        pos = rotate_pos(pos)
        centres = rotate_pos(centres)

    if sGammas is None:
        print('Getting AO gamma...', end=' ')
        start = perf_counter()
        agaL, agaR = qcm.AO_gammas(pos, gamma)
        ao_end = perf_counter()
        print(f'Done! [{ao_end - start} seconds]\nGetting sites gammas...', end=' ')
        cgamL, cgamR = qcm.MO_gammas(S, agaL, agaR, return_diag=True)
        end = perf_counter()
        print(f'Done! [{end - ao_end} seconds]')
        save_npy(cgamL, f'gamL_{run_name}.npy', npydir)
        save_npy(cgamR, f'gamR_{run_name}.npy', npydir)
        
    else:
        cgamL, cgamR = sGammas

    gamL_tol = np.mean(cgamL) + tol_scal * np.std(cgamL)
    gamR_tol = np.mean(cgamR) + tol_scal * np.std(cgamR)
    
    edArr, rdArr = diff_arrs_var_a(ee, centres, radii, eF=eF, E=E)
    
    tracker_file = f'finished_temps_{run_name}_sites_var_a.txt'
    ftrack = open(tracker_file,'w')

    if dcrits_npy:
        dcrits = np.zeros(all_Ts.shape[0])

    for k, T in enumerate(all_Ts):
        # ******* 5: Get spanning cluster *******
        conduction_clusters, dcrit, A, iidprev = percolate(ee, pos, M, T, gamL_tol=gamL_tol,gamR_tol=gamR_tol, return_adjmat=True, distance='logMA',MOgams=(cgamL, cgamR), dArrs=(edArr,rdArr))

        if dcrits_npy:
            dcrits[k] = dcrit

        
        if run_name is None:
            if hyperlocal:
                pkl_name = f'out_var_a_hl_rollback_percolate-{T}K.pkl'
            else:
                pkl_name = f'out_var_a_rollback_percolate-{T}K.pkl'
        else:
            pkl_name = 'out_percolate_' + run_name + f'-{T}K.pkl'

        with open(os.path.join(pkl_dir, pkl_name), 'wb') as fo:
            pickle.dump((conduction_clusters,dcrit,A), fo)
        ftrack.write(f'{T}K\n')
    ftrack.close()

    if dcrits_npy:
        np.save(os.path.join(pkl_dir, f'dcrits.npy'),np.vstack((all_Ts,dcrits)))


def run_percolate_from_MOs(pos, energies, M,gamL, gamR, all_Ts, eF, dV, tolscal=3.0, pkl_dir='.'):
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

        with open(os.path.join(pkl_dir, f'out_percolate-{T}K.pkl'), 'wb') as fo:
            pickle.dump((conduction_clusters,dcrit,A), fo)
        print('Saved successfully.\n')
