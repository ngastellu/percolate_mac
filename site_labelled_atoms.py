#!/usr/bin/env python

import os
import sys
import numpy as np
import pickle
from var_a_percolate_mac.percolate import diff_arrs_var_a
from scipy.spatial import KDTree


nn = int(sys.argv[1])
rmax = float(sys.argv[2])

rho = 1.05e-3
kB = 8.617e-5

outdir=f'sample-{nn}/conduction_atoms'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

sitesdir = f'sample-{nn}/new_sites_data_{rho}/'

radii = np.load(sitesdir + 'radii.npy')
S = np.load(sitesdir + 'site_state_matrix.npy')

igood = (radii < rmax).nonzero()[0]
radii = radii[igood]
S = S[:,igood]
N = S.shape[0]
nsites = S.shape[1]

labelled_atoms = (S != 0).T

np.save(f'hopping_site_masks-{nn}.npy',labelled_atoms)

 
