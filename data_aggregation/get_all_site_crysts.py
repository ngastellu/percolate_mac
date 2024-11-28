#!/usr/bin/env python

from glob import glob
import numpy as np
from sites_analysis import sites_crystallinity
import os
import sys


structype = os.path.basename(os.getcwd())


motype = sys.argv[1]
lbls = [int(d.split('-')[1]) for d in glob('sample-*')]

for n in lbls:
    print(n, end = ' --> ')

    try:
        S = np.load(f'sample-{n}/sites_data_{motype}/site_state_matrix.npy')
    except FileNotFoundError:
        print('Missing NPY for S matrix!')
        continue

    try:
        cryst_mask = np.load(os.path.expanduser(f'~/scratch/structural_characteristics_MAC/labelled_ring_centers/{structype}/sample-{n}/crystalline_atoms_mask-{n}.npy')) 
    except FileNotFoundError:
        print('Missing NPY for crystalline mask!')
        continue
     
    S /= np.linalg.norm(S,axis=0)
    site_crysts = sites_crystallinity(S,cryst_mask)
    np.save(f'sample-{n}/sites_data_{motype}/site_crystallinities.npy')
    print('Done!')