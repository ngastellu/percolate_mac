#!/usr/bin/env python

import numpy as np
from qcnico.coords_io import read_xyz
from qcnico.qcplots import plot_MO, plot_atoms
import matplotlib.pyplot as plt
from qcnico import qchemMAC as qcm
from os import path


if __name__ == '__main__':

    # Specify desired AMC structure

    structure_type = 'tempdot5'
    structure_index = 5

    if structure_type == 'tempdot5':
        rmax = 198.69
        structure_type_official = '300'
    elif structure_type == 'tempdot6':
        rmax = 121.2
        structure_type_official = 'q400'
    elif structure_type == '40x40':
        rmax = 18.03
        structure_type_official = '500'
    else:
        print('Invalid structure type.')


    # Read its atomic coordinates
    # posfile = f'structures/sAMC-{structure_type}/sAMC{structure_type}-{structure_index}.xyz' #path to the file containing the atomic positions of the structure at hand
    posfile = path.expanduser(f'~/Desktop/scripts/disorder_analysis_MAC/structures/sAMC-{structure_type_official}/sAMC{structure_type_official}-{structure_index}.xyz') #path to the file containing the atomic positions of the structure at hand
    coords = read_xyz(posfile)
    coords = coords[:,:2]

    # Load its hopping site mask
    mask_file = path.expanduser(f'~/Desktop/simulation_outputs/percolation/{structure_type}/var_radii_data/hopping_site_masks/hopping_site_masks-{structure_index}.npy') #path to the file containing the masks for each site
    # mask_file = path.expanduser(f'~/Desktop/simulation_outputs/percolation/{structure_type}/var_radii_data/strict_hopping_masks/strict_hopping_mask-{structure_index}.npy') #path to the file containing the masks for each site
    mask = np.load(mask_file)

    #Load site matrix
    sitemat_file = f'/Users/nico/Desktop/simulation_outputs/percolation/site_ket_matrices/{structure_type}_rmax_{rmax}/site_kets-{structure_index}_new.npy'
    S = np.load(sitemat_file)

    print('Shape of S: ', S.shape)
    print('Nb. of atoms: ', coords.shape[0])
    
    #Get site radii and positions
    rfile = f'/Users/nico/Desktop/simulation_outputs/percolation/{structure_type}/var_radii_data/to_local_sites_data/sample-{structure_index}/sites_data_0.00105/radii.npy'
    cfile = f'/Users/nico/Desktop/simulation_outputs/percolation/{structure_type}/var_radii_data/to_local_sites_data/sample-{structure_index}/sites_data_0.00105/centers.npy'
    efile = f'/Users/nico/Desktop/simulation_outputs/percolation/{structure_type}/var_radii_data/to_local_sites_data/sample-{structure_index}/sites_data_0.00105/ee.npy'
    ee = np.load(efile)


    radii = np.load(rfile)
    centres = np.load(cfile)

    print(radii.shape)    
    igood = radii < rmax
    radii = radii[igood]
    print(radii.shape)
    centres = centres[igood]


    # Apply mask to obtain the coods of carbon atoms belonging to one of its hopping sites
    isites = np.arange(radii.shape[0])
    rr = np.zeros_like(radii)
    cc = np.zeros_like(centres)
    for site_index in isites:
        site_atomic_coords = coords[mask[site_index,:]]
        site_ket = S[:,site_index]

        print(np.all(mask[site_index,:] == (site_ket != 0)))

        cc[site_index,:] = qcm.MO_com_hyperlocal(coords, S, site_index)
        rr[site_index] = qcm.MO_rgyr(coords, S, site_index, renormalise=True)

        print('Loaded radius = ', radii[site_index])
        # print('Computed radius = ', a)

    np.save(f'new_cc_{structure_type}-{structure_index}.npy', cc)
    np.save(f'new_rr_{structure_type}-{structure_index}.npy', rr)

    # rr = np.sort(rr)
    # radii = np.sort(radii)

    plt.scatter(rr,radii,c='r',s=10.0)
    plt.show()

    print(np.mean(np.abs(rr - radii)))
    print((rr == radii).nonzero()[0])

    isame_r = set((rr == radii).nonzero()[0])

    plt.plot(ee)
    for n in isame_r:
        plt.axvline(x=n,ymin=0,ymax=1,ls='--',lw=0.8,c='k')
    plt.show()

    isame_cc = set(np.all(cc == centres, axis=1).nonzero()[0])

    print(len(isame_cc))
    print(len(isame_r))
    print(isame_cc - isame_r)
    print(isame_r < isame_cc)