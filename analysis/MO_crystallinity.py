#!/usr/bin/env python

import numpy as np
import sys
import os
from sites_analysis import undistorted_mask


def MOs_crystallinity(M,cryst_mask,cryst_renormalise=False):
    psi_cryst = M[cryst_mask,:] # only AOs on crystalline sites
    if cryst_renormalise:
        return (np.abs(psi_cryst)**2).sum(axis=0) / cryst_mask.sum()
    else:
        return (np.abs(psi_cryst)**2).sum(axis=0)


if __name__ == '__main__':

    ensemble = sys.argv[1]
    cryst_mask_type = sys.argv[2]
    psipow=2
    motype = 'virtual'
    renormalise = True
    if renormalise:
        outdir = f'MO_crystallinities_{cryst_mask_type}_{ensemble}_renormd/'
    else:
        outdir = f'MO_crystallinities_{cryst_mask_type}_{ensemble}/'

    if not os.path.exists(outdir):
        os.mkdir(outdir)


    if ensemble == '40x40':
        lbls = np.arange(1,300)
    elif ensemble == 'tempdot6':
        lbls = np.arange(132)
    elif ensemble == 'tempdot5':
        lbls = np.arange(117)
    else:
        print(f'{ensemble} is an invalid ensemble name.')


    for n in lbls:
        
        try:
            M = np.load(os.path.expanduser(f'~/scratch/ArpackMAC/{ensemble}/MOs/{motype}/MOs_ARPACK_bigMAC-{n}.npy'))
            N = M.shape[0]
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
        
        
        electronic_crystallinity = MOs_crystallinity(M,cryst_mask,renormalise)

        np.save(outdir + f'MO_crystallinities-{n}.npy', electronic_crystallinity)