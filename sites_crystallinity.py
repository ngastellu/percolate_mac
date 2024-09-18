#!/usr/bin/env python

import numpy as np
from MO_crystallinity import MOs_crystallinity
from sites_analysis import undistorted_mask
import sys
import os



ensemble = sys.argv[1]
cryst_mask_type = sys.argv[2]
psipow=2
eps_rho = 0.00105
motype = 'virtual'
renormalise = False

if ensemble == '40x40':
    lbls = np.arange(1,300)
    rmax = 18.03
elif ensemble == 'tempdot6':
    lbls = np.arange(132)
    rmax = 121.2
elif ensemble == 'tempdot5':
    lbls = np.arange(117)
    rmax = 198.69
else:
    print(f'{ensemble} is an invalid ensemble name.')


sites_datadir = f'sites_data_{eps_rho}_psi_pow{psipow}/'


for n in lbls:  
    if renormalise:
        outdir = f'sample-{n}/sites_crystallinities_{cryst_mask_type}_{ensemble}_renormd_by_nb_cryst_atoms/'
    else:
        outdir = f'sample-{n}/sites_crystallinities_{cryst_mask_type}_{ensemble}/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        os.mkdir(outdir + 'all_sites/')
        os.mkdir(outdir + f'rmax_{rmax}/')

    try:
        S = np.load(f'sample-{n}/' + sites_datadir + 'site_state_matrix.npy')
        S /= np.linalg.norm(S,axis=0)
        radii = np.load(f'sample-{n}/' + sites_datadir + 'radii.npy')
        N = S.shape[0]
    except FileNotFoundError as e:
        print(e)
        continue

    if cryst_mask_type == 'crystalline':
        try:
            cryst_mask_dir = os.path.expanduser(f'~/scratch/structural_characteristics_MAC/labelled_ring_centers/{ensemble}/sample-{n}/')
            cryst_mask = np.load(cryst_mask_dir +  f'crystalline_atoms_mask-{n}.npy')
        except FileNotFoundError as e:
            print(e)
            continue
    elif cryst_mask_type == 'undistorted':
        try:
            cryst_mask_dir = os.path.expanduser(f'~/scratch/structural_characteristics_MAC/rho_sites/{ensemble}/sample-{n}/')
            undistorted_atoms = np.load(cryst_mask_dir +  f'undistorted_atoms_{ensemble}-{n}.npy')
            cryst_mask = undistorted_mask(undistorted_atoms,N)
        except FileNotFoundError as e:
            print(e)
            continue
    else:
        print(f'{cryst_mask_type} is an invalid crystalline mask type.')
        sys.exit()
    
    
    electronic_crystallinity = MOs_crystallinity(S,cryst_mask,renormalise)
    np.save(outdir + f'all_sites/sites_crystallinities-{n}.npy', electronic_crystallinity)

    rfilter = radii < rmax
    S = S[:,rfilter]
    electronic_crystallinity = MOs_crystallinity(S,cryst_mask,renormalise) 
    np.save(outdir + f'rmax_{rmax}/sites_crystallinities-{n}.npy', electronic_crystallinity)
