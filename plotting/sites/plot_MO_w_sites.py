#!/usr/bin/env python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from qcnico.qchemMAC import MO_com, MO_rgyr
from qcnico.qcplots import plot_MO
from qcnico.coords_io import read_xyz



structype = 'tempdot5'

if structype == 'tempdot5':
    off_structype = 'sAMC-300'
elif structype == 'tempdot6':
    off_structype = 'sAMC-q400'
elif structype == '40x40':
    off_structype = 'sAMC-500'
else:
    print(f'Structure type {structype} is invalid! Exiting with error.')
    sys.exit()

istruc = 85
motype = 'lo'

simdir = os.path.expanduser('~/Desktop/simulation_outputs')

posdir = os.path.join(simdir, f'MAC_structures/relaxed_no_dangle/{off_structype}')
pos  = read_xyz(os.path.join(posdir, f'{('').join(off_structype.split('-'))}-{istruc}.xyz'))
pos = pos[:,:2]

M = np.load(os.path.join(simdir, f'percolation/{structype}/MOs_ARPACK/{motype}/MOs_ARPACK_{motype}_{structype}-{istruc}.npy'))

sitesdir = os.path.join(simdir, f'percolation/{structype}/var_radii_data/to_local_sites_data_0.00105_psi_pow2_lo/sample-{istruc}')

S = np.load(os.path.join(sitesdir, 'site_state_matrix.npy'))
radii = np.load(os.path.join(sitesdir, 'radii.npy')) 
centers = np.load(os.path.join(sitesdir, 'centers.npy')) 
ii = np.load(os.path.join(sitesdir, 'ii.npy'))


# np.random.seed(42)
# for iMO in np.random.randint(M.shape[1], size=10):
iMO = 93
print(iMO)
isites = (ii == iMO).nonzero()[0]
fig, axs = plt.subplots(1,2,sharey=True)
fig, axs[0] = plot_MO(pos, M, iMO, dotsize=0.5,loc_centers=centers[isites], loc_radii=radii[isites],plt_objs=(fig,axs[0]),show=False, show_cbar=False)
fig, axs[1] = plot_MO(pos, M, iMO, dotsize=0.5,loc_centers=MO_com(pos, M, iMO), loc_radii=[MO_rgyr(pos,M,iMO)],plt_objs=(fig,axs[1]),show=False, show_cbar=True)
plt.show()
