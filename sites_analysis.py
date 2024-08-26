#!/usr/bin/env python


from os import path
import sys
import numpy as np
from var_a_percolate_mac.utils_analperc import get_conduction_clusters, conduction_mask


def undistorted_mask(iundistorted,N):
    """Converts a list of indices to a boolean mask of size n."""
    
    out = np.zeros(N, dtype=bool)
    out[iundistorted] = True
    return out


def mask_overlap(mask1, mask2, return_bools=True):
    """Computes the AND (i.e. intersection) between two boolean masks."""
    
    if return_bools:
        return mask1 * mask2
    else:
        return (mask1 * mask2).nonzero()[0]


def track_overlaps(datadir, structural_mask, sites_mask, temps, perc_run_name, mask_name):
    """Computes how many conduction atoms belong to a crystallite (or other structural feature as indicated by `structural_mask`) as a function of temperature."""
    N = structural_mask.shape[0]
    pkl_prefix = f'out_percolate_{perc_run_name}'
    for T in temps:
        print(f' -- T = {T} K --')
        clusters = get_conduction_clusters(datadir, pkl_prefix, T)
        if len(clusters) > 1:
            print(f'Multiple ({len(clusters)}) conduction clusters detected!', flush=True)
            print('Cluster sizes = ', [len(c) for c in clusters],flush=True)
            clusters_mask = [conduction_mask(sites_mask,c) for c in clusters]
            overlaps = np.array([mask_overlap(structural_mask,cm, return_bools=True) for cm in clusters_mask])
        else:
            cluster_mask = conduction_mask(sites_mask, clusters[0])
            overlaps = mask_overlap(structural_mask,cluster_mask,return_bools=True)
        
        np.save(path.join(datadir, f'{mask_name}_conduction_masks_{perc_run_name}-{T}K.npy'), overlaps)


if __name__ == "__main__":

    structype = sys.argv[1]
    struct_mask_type = sys.argv[2]
    site_mask_type = sys.argv[3]

    if structype == '40x40':
        rmax = 18.03
    elif structype == 'tempdot6':
        rmax = 121.2
    elif structype == 'tempdot5':
        rmax = 198.69
    else:
        print(f'{structype} is an invalid structure type!')
        sys.exit()

    run_lbl = f'rmax_{rmax}_psipow1'
    pkl_prefix = f'out_percolate_{run_lbl}'
    temps = np.arange(40,440,10)


    
    with open(f'to_local_{run_lbl}/good_runs_{run_lbl}.txt') as fo:
        lines = fo.readlines()

    nn = [int(l.strip()) for l in lines]

    for n in nn:
        print(f'****** Doing sample {n} ******')
        try:
            datadir = f'sample-{n}' # run this from the percolation dir

            if site_mask_type == 'naive':
                site_mask = np.load(f'hopping_site_masks/hopping_site_masks-{n}.npy')
            elif site_mask_type == 'strict':
                site_mask = np.load(f'strict_hopping_masks/strict_hopping_site-{n}.npy')
            else:
                print(f'Invalid specified site_mask_type: {site_mask_type}')
                sys.exit()

            N = site_mask.shape[1]

            if struct_mask_type == 'undistorted':
                struc_mask_datadir = path.expanduser(f'~/scratch/structural_characteristics_MAC/rho_sites/{structype}/')
                iundistorted = np.load(path.join(struc_mask_datadir,f'sample-{n}',f'undistorted_atoms_{structype}-{n}.npy'))
                structural_mask = undistorted_mask(iundistorted, N)
            elif struct_mask_type == 'crystalline':
                struc_mask_datadir = path.expanduser(f'~/scratch/structural_characteristics_MAC/labelled_ring_cen/{structype}/sample-{n}/')
                structural_mask = np.load(path.join(struc_mask_datadir, f'crystalline_atoms_mask-{n}.npy'))
            else:
                print(f'Invalid specified struct_mask_type: {struct_mask_type}')
                sys.exit()
            


            track_overlaps(datadir, structural_mask, site_mask, temps, run_lbl, struct_mask_type)
        
        except Exception as e:
            print(e)

