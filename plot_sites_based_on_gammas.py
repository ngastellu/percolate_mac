#!/usr/bin/env python

import numpy as np
from qcnico.plt_utils import setup_tex
from qcnico.qcplots import plot_MO 
from qcnico.coords_io import read_xyz
import os
import sys


np.random.seed(64)

structype = 'tempdot5'

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

istruc = 1
istate = 0
motype = 'lo'
apply_rfilter = True
tolscal = 3.0



simdir = os.path.expanduser('~/Desktop/simulation_outputs')

posdir = os.path.join(simdir, f'MAC_structures/relaxed_no_dangle/{off_structype}')
pos  = read_xyz(os.path.join(posdir, f'{('').join(off_structype.split('-'))}-{istruc}.xyz'))

sitesdir = os.path.join(simdir, f'percolation/{structype}/var_radii_data/sites_data_{motype}/sample-{istruc}')

S = np.load(os.path.join(sitesdir, 'site_state_matrix.npy'))
S /= np.linalg.norm(S,axis=0)

radii = np.load(os.path.join(sitesdir, 'radii.npy')) 
centers = np.load(os.path.join(sitesdir, 'centers.npy')) 
ii = np.load(os.path.join(sitesdir, 'ii.npy'))
gamL = np.load(os.path.join(sitesdir, f'gamL_rmax_{rmax}_sites_gammas_{motype}.npy'))
gamR = np.load(os.path.join(sitesdir, f'gamR_rmax_{rmax}_sites_gammas_{motype}.npy'))

gamL_tol = np.mean(gamL) + tolscal * np.std(gamL)
gamR_tol = np.mean(gamR) + tolscal * np.std(gamR)

if apply_rfilter:
    rfilter = radii < rmax 
    radii = radii[rfilter]
    centers = centers[rfilter]
    ii = ii[rfilter]
    S = S[:,rfilter]
    # !! gammas are already rfiltered !!
Sbinary = (S != 0).astype(int)

igamL = (gamL > gamL_tol).nonzero()[0]
igamR = (gamR > gamR_tol).nonzero()[0]

print('# of L-coupled sites = ', igamL.shape)
print('# of R-coupled sites = ', igamR.shape)

# plot_inds_L = np.random.choice(igamL,size=3)
# plot_inds_R = np.random.choice(igamR,size=3)
# plot_inds = np.hstack((plot_inds_L,plot_inds_R))
# for i in plot_inds:
#     plot_MO(pos, S, i, dotsize=0.5,loc_centers=np.array([centers[i]]), loc_radii=[radii[i]],scale_up=10)
#     plot_MO(pos, Sbinary, i, dotsize=0.5,loc_centers=np.array([centers[i]]), loc_radii=[radii[i]],cmap='bwr')

plot_MO(pos, S, igamL, dotsize=0.5, loc_centers=centers[igamL], loc_radii=radii[igamL], scale_up=10)
plot_MO(pos, S, igamR, dotsize=0.5, loc_centers=centers[igamR], loc_radii=radii[igamR], scale_up=10)