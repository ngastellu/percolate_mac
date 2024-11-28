#!/usr/bin/env python

import numpy as np
from qcnico.plt_utils import setup_tex
from qcnico.qcplots import plot_MO
from qcnico.coords_io import read_xyz
import os
import sys




structype = 'tempdot5'

if structype == 'tempdot5':
    synth_temp = '300'
elif structype == 'tempdot6':
    synth_temp = 'q400'
elif structype == '40x40':
    synth_temp = '500'
else:
    print(f'Structure type {structype} is invalid! Exiting with error.')
    sys.exit()

istruc = 1
# istate = 123
motype = 'lo'



simdir = os.path.expanduser('~/Desktop/simulation_outputs')

posdir = os.path.join(simdir, f'MAC_structures/relaxed_no_dangle/sAMC-{synth_temp}')
pos  = read_xyz(os.path.join(posdir, f'sAMC{synth_temp}-{istruc}.xyz'))

sitesdir = os.path.join(simdir, f'percolation/{structype}/var_radii_data/sites_data_{motype}/sample-{istruc}')

S = np.load(os.path.join(sitesdir, 'site_state_matrix.npy'))
radii = np.load(os.path.join(sitesdir, 'radii.npy')) 
centers = np.load(os.path.join(sitesdir, 'centers.npy')) 
ii = np.load(os.path.join(sitesdir, 'ii.npy'))

istate = np.argmax(radii)

MO_index = ii[istate]
print(f'MO index of site {istate} =  {MO_index}')
print(f'Other sites from the MO # {MO_index}:')
for nn in (ii == MO_index).nonzero()[0]:
    print(nn)

isiblings = (ii == MO_index).nonzero()[0]
print(len(isiblings))

for i in isiblings:
    plot_MO(pos, S, i, dotsize=0.5,loc_centers=np.array([centers[i]]), loc_radii=[radii[i]])