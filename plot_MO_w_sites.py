#!/usr/bin/env python

import sys
import os
import numpy as np
from qcnico.qcplots import plot_MO
from qcnico.coords_io import read_xyz



structype = 'tempdot6'

if structype == 'tempdot5':
    off_structype = 'sAMC-300'
elif structype == 'tempdot6':
    off_structype = 'sAMC-q400'
elif structype == '40x40':
    off_structype = 'sAMC-500'
else:
    print(f'Structure type {structype} is invalid! Exiting with error.')
    sys.exit()

istruc = 69
motype = 'lo'

simdir = os.path.expanduser('~/Desktop/simulation_outputs')

posdir = os.path.join(simdir, f'MAC_structures/relaxed_no_dangle/{off_structype}')
pos  = read_xyz(os.path.join(posdir, f'{('').join(off_structype.split('-'))}-{istruc}.xyz'))

M = np.load(os.path.join(simdir, f'percolation/{structype}/MOs_ARPACK/{motype}/MOs_ARPACK_{motype}_{structype}-{istruc}.npy'))

sitesdir = os.path.join(simdir, f'percolation/{structype}/var_radii_data/to_local_sites_data_0.00105_psipow2_lo/sample-{istruc}')

S = np.load(os.path.join(sitesdir, 'site_state_matrix.npy'))
radii = np.load(os.path.join(sitesdir, 'radii.npy')) 
centers = np.load(os.path.join(sitesdir, 'centers.npy')) 
ii = np.load(os.path.join(sitesdir, 'ii.npy'))


np.random.seed(42)
for iMO in np.random.randint(M.shape[1], size=10):
    isites = (ii == iMO).nonzero()[0]
    plot_MO(pos, M, iMO, dotsize=0.5,loc_centers=centers[isites], loc_radii=radii[isites])
