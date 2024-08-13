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
    structure_index = 12

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
    mask_file = path.expanduser(f'~/Desktop/simulation_outputs/percolation/{structure_type}/var_radii_data/strict_hopping_masks/strict_hopping_mask-{structure_index}.npy') #path to the file containing the masks for each site
    mask = np.load(mask_file)

    #Load site matrix
    # sitemat_file = f'/Users/nico/Desktop/simulation_outputs/percolation/site_ket_matrices/{structure_type}_rmax_{rmax}/site_kets-{structure_index}_newestest.npy'
    # S = np.load(sitemat_file)

    # sitemat_file = f'/Users/nico/Desktop/simulation_outputs/percolation/site_ket_matrices/{structure_type}_rmax_{rmax}/site_kets-{structure_index}_new.npy'
    # S2 = np.load(sitemat_file)

    
    #Get site radii and positions
    # rfile = f'/Users/nico/Desktop/simulation_outputs/percolation/{structure_type}/var_radii_data/to_local_sites_data/sample-{structure_index}/sites_data_0.00105/radii.npy'
    # cfile = f'/Users/nico/Desktop/simulation_outputs/percolation/{structure_type}/var_radii_data/to_local_sites_data/sample-{structure_index}/sites_data_0.00105/centers.npy'
    # lfile = f'/Users/nico/Desktop/simulation_outputs/percolation/{structure_type}/var_radii_data/to_local_sites_data/sample-{structure_index}/sites_data_0.00105/labels.npy'
    sites_datadir = f'/Users/nico/Desktop/simulation_outputs/percolation/tempdot5/var_radii_data/sites_data_0.00105_psi_pow1-12/'
    rfile = sites_datadir + 'radii.npy'
    cfile = sites_datadir + 'centers.npy'
    lfile = sites_datadir + 'labels.npy'
    sitemat_file = sites_datadir + 'site_state_matrix.npy'

    radii = np.load(rfile)
    centres = np.load(cfile)
    S = np.load(sitemat_file)

    print(radii.shape)    
    igood = radii < rmax
    radii = radii[igood]
    print(radii.shape)
    centres = centres[igood]
    S = S[:,igood]
    print('Shape of S: ', S.shape)
    print('Nb. of atoms: ', coords.shape[0])
    # S2 = S2[:,igood]


    # Apply mask to obtain the coods of carbon atoms belonging to one of its hopping sites
    # isites = [0,1]
    # isites=[389, 139, 396, 13, 14, 397, 16, 273, 18, 274]
    np.random.seed(0)
    # isites = np.random.randint(S.shape[1],size=10)
    # isites = [417, 362, 381,  71, 380, 159, 195, 190,  73, 187] #max rdiffs for sAMC300-12
    isites = [199, 387, 181, 159, 109,  63, 233,  42, 227, 106] #max center dists for sAMC300-12
    for site_index in isites:
        print(f'\n\n-------- {site_index} --------')
        # site_atomic_coords = coords[mask[site_index,:]]

        site_ket = S[:,site_index]
        # site_ket2 = S2[:,site_index]

        # print(np.all(mask[site_index,:] == (site_ket != 0)))

        r = qcm.MO_com_hyperlocal(coords, S, site_index)
        # r2 = qcm.MO_com_hyperlocal(coords, S2, site_index)


        print('Loaded centre = ', centres[site_index])
        print('Computed centre = ', r)

        a = qcm.MO_rgyr(coords, S, site_index, renormalise=True)
        # a2 = qcm.MO_rgyr(coords, S2, site_index, renormalise=True)

        print('Loaded radius = ', radii[site_index])
        print('Computed radius = ', a)

        plot_MO(coords, S, site_index, dotsize=1.0, loc_centers=np.array([centres[site_index], r]), loc_radii=[radii[site_index], a],scale_up=10.0, c_clrs=['r', 'g'])
        # plot_MO(coords, S2, site_index, dotsize=1.0, loc_centers=np.array([centres[site_index], r2]), loc_radii=[radii[site_index], a2],scale_up=10.0, c_clrs=['r', 'g'])



        # colours = np.array(['k']* coords.shape[0])
        # colours[mask[site_index,:]] = 'r'

        # plot_atoms(coords,dotsize=1.0,colour=colourst
