#!/usr/bin/env python

import numpy as np
from qcnico.coords_io import read_xyz
from qcnico.qcplots import plot_MO

nn = 15
structype = 'tempdot6'

sites_datadir = f'/Users/nico/Desktop/simulation_outputs/percolation/{structype}/var_radii_data/to_local_sites_data/sample-{nn}/sites_data_0.00105/'
radii = np.load(sites_datadir + 'radii.npy')

ibigs = np.argsort(radii)[-2:]

S = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/site_ket_matrices/{structype}_rmax_121.2/site_kets_psipow2-{nn}.npy')
S /= np.linalg.norm(S,axis=0)

#big_ket = S[:,ibig]
#big_ket /= np.linalg.norm(big_ket)

for ibig in ibigs:

    center = np.load(sites_datadir + 'centers.npy')[ibig]
    radius = radii[ibig]
    MO_inds= np.load(sites_datadir + 'ii.npy')
    iMO = MO_inds[ibig]
    print(f'Nb. of sibling sites (i.e. also from MO nb. {iMO}) = ', (MO_inds == iMO).sum())

    pos = read_xyz(f'/Users/nico/Desktop/simulation_outputs/MAC_structures/relaxed_no_dangle/sAMC-q400/sAMCq400-{nn}.xyz')

    plot_MO(pos, S, ibig, dotsize=1.0,scale_up=10,loc_centers=np.array([center]),loc_radii=np.array([radius]), c_rel_size=50,c_lw=2.0)