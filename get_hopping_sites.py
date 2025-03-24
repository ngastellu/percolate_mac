#!/usr/bin/env python

import os
import sys
import numpy as np
from deploy_percolate import VRH_sites


n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_type = sys.argv[3]

arpackdir = os.path.expanduser('~/scratch/ArpackMAC/')
xyzdir = os.path.expanduser('~/scratch/clean_bigMAC/final_sAMC_structures/')

dod_temp = struc_type.split('-')[1]
xyz_prefix = f'sAMC{dod_temp}-'

centres, radii, site_energies, MO_indices, labels, site_matrix = VRH_sites(n, struc_type, mo_type, xyzdir, arpackdir, xyz_prefix)

npydir = f'sites_data_{mo_type}' 
if not os.path.isdir(npydir):
    os.mkdir(npydir)

np.save(os.path.join(npydir,'ee.npy'), site_energies)
np.save(os.path.join(npydir, 'radii.npy'), radii)
np.save(os.path.join(npydir, 'centres.npy'), centres)
np.save(os.path.join(npydir, 'ii.npy'), MO_indices)
#    np.save(npydir + 'rr_v_masses_v_iprs_v_ee.npy',np.vstack((r_sorted,masses,site_iprs,e_sorted)))
np.save(os.path.join(npydir, 'site_state_matrix.npy'),site_matrix)


