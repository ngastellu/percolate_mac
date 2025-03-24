#!/usr/bin/env python

import sys
from os import path
import numpy as np
from scipy.spatial import KDTree
from qcnico.coords_io import read_xyz





structype = sys.argv[1] 
nn = int(sys.argv[2])
eps_rho = 0.00105

if structype == '40x40':
    xyz_prefix = 'bigMAC-'
    rmax = 18.03
else:
    xyz_prefix = structype + 'n'
    if structype == 'tempdot5':
        rmax = 198.96
    else:
        rmax = 121.2

pos = read_xyz(path.expanduser(f'~/scratch/clean_bigMAC/{structype}/relaxed_structures_no_dangle/{xyz_prefix}{nn}_relaxed_no-dangle.xyz'))
pos = pos[:,:2]
N = pos.shape[0]

radii = np.load(f'sample-{nn}/sites_data_{eps_rho}/radii.npy')
centres = np.load(f'sample-{nn}/sites_data_{eps_rho}/centers.npy')

rselect = radii < rmax
radii = radii[rselect]
centres = centres[rselect]

nsites = radii.shape[0]
mask = np.zeros((nsites,N),dtype=bool)

tree = KDTree(pos)


for n in range(nsites):
    cc = centres[n]
    r = radii[n]
    iatoms = tree.query_ball_point(cc,r)
    mask[n,iatoms] = True

np.save(f'strict_hopping_masks/strict_hopping_mask-{nn}.npy', mask)