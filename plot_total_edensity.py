
#!/usr/bin/env python 

import numpy as np
from qcnico.plt_utils import setup_tex
from qcnico.qcplots import plot_MO
from qcnico.coords_io import read_xyz
import os
import sys




structype = 'tempdot5'
motype = 'lo'

if structype == 'tempdot5':
    off_structype = 'sAMC-300'
    rmax = 198.69
elif structype == 'tempdot6':
    off_structype = 'sAMC-q400'
    rmax = 121.2
elif structype == '40x40':
    off_structype = 'sAMC-500'
    rmax = 18.03
else:
    print(f'Structure type {structype} is invalid! Exiting with error.')
    sys.exit()

istruc = 99



simdir = os.path.expanduser('~/Desktop/simulation_outputs')

posdir = os.path.join(simdir, f'MAC_structures/relaxed_no_dangle/{off_structype}')
pos  = read_xyz(os.path.join(posdir, f'{('').join(off_structype.split('-'))}-{istruc}.xyz'))

if motype == 'virtual':
    sitesdir = os.path.join(simdir, f'percolation/{structype}/var_radii_data/to_local_sites_data_0.00105_psi_pow2/sample-{istruc}')
    # M = np.load(os.path.join(simdir, f"percolation/{structype}/MOs_ARPACK/{motype}/MOs_ARPACK_bigMAC-{istruc}.npy"))
else:
    sitesdir = os.path.join(simdir, f'percolation/{structype}/var_radii_data/to_local_sites_data_0.00105_psi_pow2_{motype}/sample-{istruc}')
    # M = np.load(os.path.join(simdir, f"percolation/{structype}/MOs_ARPACK/{motype}/MOs_ARPACK_{motype}_{structype}-{istruc}.npy"))

M = np.load(os.path.join(sitesdir, 'site_state_matrix.npy'))
# radii = np.load(os.path.join(sitesdir, 'radii.npy'))
# M = M[:, radii >= rmax]
plot_MO(pos, M, np.arange(M.shape[1]), dotsize=0.5,scale_up=10.0)