#!/usr/bin/env python

import sys
import numpy as np
import os
from sites_analysis import cluster_crystallinity, undistorted_mask


structype = sys.argv[1]
struct_mask_type = 'crystalline'
eps_rho = 0.00105
psipow = 2
cryst_renormalise = True

if structype == '40x40':
    rmax = 18.03
elif structype == 'tempdot6':
    rmax = 121.2
elif structype == 'tempdot5':
    rmax = 198.69
else:
    print(f'{structype} is an invalid structure type!')
    sys.exit()

run_lbl = f'rmax_{rmax}'
pkl_prefix = f'out_percolate_{run_lbl}'
temps = np.arange(40,440,10)



with open(f'to_local_{run_lbl}/good_runs_{run_lbl}.txt') as fo:
    lines = fo.readlines()

nn = [int(l.strip()) for l in lines]

for n in nn:
    print(f'****** Doing sample {n} ******')

    # Load site kets, filter them according to radius criterion, and renormalise
    datadir = f'sample-{n}/'
    S = np.load(datadir + f'sites_data_{eps_rho}_psi_pow{psipow}/site_state_matrix.npy')
    radii = np.load(datadir + f'sites_data_{eps_rho}_psi_pow{psipow}/radii.npy')
    igood = (radii < rmax)
    S = S[:,igood] # sites in cluster are indexed only from this filtered set of sites
    S /= np.linalg.norm(S,axis=0) # ensure site kets are normalised
    N = S.shape[0]

    # Load site masks
    if struct_mask_type == 'undistorted':
        struc_mask_datadir = os.path.expanduser(f'~/scratch/structural_characteristics_MAC/rho_sites/{structype}/')
        iundistorted = np.load(os.path.join(struc_mask_datadir,f'sample-{n}',f'undistorted_atoms_{structype}-{n}.npy'))
        structural_mask = undistorted_mask(iundistorted, N)
    elif struct_mask_type == 'crystalline':
        struc_mask_datadir = os.path.expanduser(f'~/scratch/structural_characteristics_MAC/labelled_ring_centers/{structype}/sample-{n}/')
        structural_mask = np.load(os.path.join(struc_mask_datadir, f'crystalline_atoms_mask-{n}.npy'))
    else:
        print(f'Invalid specified struct_mask_type: {struct_mask_type}')
        sys.exit()
    

    if cryst_renormalise:
        outdir = datadir + f'cluster_crystallinities_{run_lbl}_renormd/'
    else:
        outdir = datadir + f'cluster_crystallinities_{run_lbl}/'
        
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for T in temps:
        print(T,end = ' ')
        cl_cryst = cluster_crystallinity(datadir,structural_mask,S,T,run_lbl,renormalise_by_cryst_size=cryst_renormalise)
        np.save(outdir + f'clust_cryst-{T}K.npy', cl_cryst)
        