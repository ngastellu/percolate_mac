#!/usr/bin/env python

import pickle
from time import perf_counter
import os
import numpy as np
import qcnico.qchemMAC as qcm
from qcnico.coords_io import read_xyz
from qcnico.rotate_pos import rotate_pos
from qcnico.data_utils import save_npy
from percolate import percolate, diff_arrs_var_a
from MOs2sites import generate_sites_radii_list, LR_MOs

def load_atomic_positions(n, xyzdir, xyz_prefix, xyz_suffix=''):
  if len(xyz_suffix) >= 3 and  xyz_suffix[-3] == '.xyz':
    filename = f'{xyz_prefix}{n}{xyz_suffix}'
  else:
    filename = f'{xyz_prefix}{n}{xyz_suffix}.xyz'

  pos, symbols = read_xyz(os.path.join(xyzdir,filename), filename)
  return pos



def load_arpack_data(pos, sample_index, structype, motype,arpackdir,gamma_dir='.',gammas_method='compute',sort_energies=False):
    """ Loads energies, MOs, and coupling matrices of a given MAC structure.
    This function aims to be be common to all percolation runs (gridMOs or not, etc.). """

    arpackdir = os.path.join(arpackdir, structype)

    
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
     
    # ******* 2: Get gammas *******
    if gammas_method == 'compute':
        gamma = 0.1
        agaL, agaR = qcm.AO_gammas(pos, gamma)
        gamL, gamR = qcm.MO_gammas(M, agaL, agaR, return_diag=True)
        np.save(os.path.join(gamma_dir,f'gamL_40x40-{sample_index}_{motype}.npy'), gamL)
        np.save(os.path.join(gamma_dir,f'gamR_40x40-{sample_index}_{motype}.npy'), gamR)
        
        return energies, M, gamL, gamR

    elif gammas_method == 'load':
        try:
            gamL = np.load(os.path.join(gamma_dir, f'gamL_40x40-{sample_index}_{motype}.npy'))
            gamR = np.load(os.path.join(gamma_dir, f'gamR_40x40-{sample_index}_{motype}.npy'))
        except FileNotFoundError:
            print('Gamma files not found. Re-computing gammas.')
            gamma = 0.1
            agaL, agaR = qcm.AO_gammas(pos, gamma)
            gamL, gamR = qcm.MO_gammas(M, agaL, agaR, return_diag=True)
            np.save(os.path.join(f'gamL_40x40-{sample_index}_{motype}.npy'), gamL)
            np.save(os.path.join(f'gamR_40x40-{sample_index}_{motype}.npy'), gamR)
    
        return energies, M, gamL, gamR
    elif gammas_method == 'none':
        return energies, M
    else:
        print(f'''[load_data ERROR] Invalid for entry for `gammas_method`: {gammas_method}.
              \nValid entries are:\n
              * "compute" (default): Computes all of the MO-lead couplings from scratch\n
              * "load": Checks if MO-lead couplings have already been computed in saved in current directory. If yes, load them from NPY files. If no, computes them from scratch (same as "compute")\n
              * "none": does not retrieve the couplings; returns only pos, energies, and MO matrix.\n
            Assuming "none" here.
              ''')
        return energies, M



def load_var_a_data(datadir='.'):
    sites_pos = np.load(os.path.join(datadir, 'centers.npy'))
    sites_energies = np.load(os.path.join(datadir, 'ee.npy'))
    sites_radii = np.load(os.path.join(datadir, 'radii.npy'))
    ii = np.load(os.path.join(datadir, 'ii.npy'))

    return sites_pos, sites_energies, sites_radii, ii


def VRH_sites(n, struc_type, mo_type, xyzdir, arpackdir, xyz_prefix):

    pos = load_atomic_positions(n,xyzdir,xyz_prefix)
    energies, M, gamL, gamR = load_arpack_data(pos, n, struc_type, mo_type, arpackdir, gammas_method='compute', sort_energies=True)
    N = pos.shape[0]

    # Get rid of HOMO when generating sites; we only include it in the MOs and energies to readily compute eF
    if mo_type == 'virtual_w_HOMO' and (N%2==0):
        energies = energies[1:]
        M = M[:,1:]
        gamL = gamL[1:]
        gamR = gamR[1:]

    pos = pos[:,:2]

    L, R = LR_MOs(gamL, gamR)


    eps_rho = 0.0
    flag_empty_clusters = True
    max_r = 50.0

    print('Generating sites and radii now...')
    start = perf_counter()
    centres, radii, site_energies, ii,labels, site_matrix = generate_sites_radii_list(pos, M, L, R, energies, radii_rho_threshold=eps_rho,flag_empty_clusters=flag_empty_clusters,max_r=max_r,return_labelled_atoms=True,return_site_matrix=True, amplitude_pow=2)
    end = perf_counter()
    print(f'Done! [{end-start} seconds]')

    return centres, radii, site_energies, ii, labels, site_matrix



def run_percolate(pos, centres, ee, radii, S, all_Ts, dV,
                   eF=0, rotate=False,
                   gamma=0.1, tol_scal=3.0, sGammas=None,
                   rmax=None, dE_max=None, 
                   run_name=None, pkl_dir='.',dcrits_npy=True):
    
    if not os.path.isdir(pkl_dir):
        os.mkdir(pkl_dir)

    # Ensure site kets are properly normalised
    S /= np.linalg.norm(S,axis=0)


    # ******* Pre-compute distances *******
    
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

    print('[run_percolate] Nsites = ', centres.shape[0])

    np.save(os.path.join(pkl_dir, f'igood_{run_name}.npy'), igood)
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
        save_npy(cgamL, f'gamL_{run_name}.npy')
        save_npy(cgamR, f'gamR_{run_name}.npy')
        
    else:
        cgamL, cgamR = sGammas

    gamL_tol = np.mean(cgamL) + tol_scal * np.std(cgamL)
    gamR_tol = np.mean(cgamR) + tol_scal * np.std(cgamR)
    
    edArr, rdArr, _ = diff_arrs_var_a(ee, centres, radii, eF=eF, E=E)
    
    tracker_file = f'finished_temps_{run_name}_sites_var_a.txt'
    ftrack = open(tracker_file,'w')

    if dcrits_npy:
        dcrits = np.zeros(all_Ts.shape[0])

    for k, T in enumerate(all_Ts):
        # ******* 5: Get spanning cluster *******
        conduction_clusters, dcrit, A, _ = percolate(edArr, rdArr, T, cgamL, cgamR, gamL_tol, gamR_tol, return_adjmat=True)

        if dcrits_npy:
            dcrits[k] = dcrit

        
        if run_name is None:
            pkl_name = f'out_percolate-{T}K.pkl'
        else:
            pkl_name = 'out_percolate_' + run_name + f'-{T}K.pkl'

        with open(os.path.join(pkl_dir, pkl_name), 'wb') as fo:
            pickle.dump((conduction_clusters,dcrit,A), fo)
        ftrack.write(f'{T}K\n')
    ftrack.close()

    if dcrits_npy:
        np.save(os.path.join(pkl_dir, f'dcrits.npy'), np.vstack((all_Ts,dcrits)))

