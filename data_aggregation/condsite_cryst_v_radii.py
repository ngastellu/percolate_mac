#!/usr/bin/env python

import numpy as np
import os
import sys
from qcnico.data_utils import save_npy
from sites_analysis import cluster_crystallinity


structype = os.path.basename(os.getcwd())

if structype == '40x40':
    rmax = 18.03
elif structype == 'tempdot6':
    rmax = 121.2
elif structype == 'tempdot5':
    rmax = 198.69


motype = sys.argv[1]
run_name = f'rmax_{rmax}_sites_gammas_{motype}'

with open(f'{run_name}_symlinks/good_runs_{run_name}.txt') as fo:
    lbls = [int(l.strip()) for l in fo]

temps = [180, 300, 430]

for n in lbls:
    outdir = f'condsites_crysts_v_radii/{run_name}/'

    print(f'{n}:', end = ' ')
    perc_dir = f'sample-{n}/{run_name}_pkls/'
    sites_dir = f'sample-{n}/sites_data_{motype}'

    cryst_mask = np.load(os.path.expanduser(f'~/scratch/structural_characteristics_MAC/labelled_ring_centers/{structype}/sample-{n}/crystalline_atoms_mask-{n}.npy'))

    S = np.load(os.path.join(sites_dir, 'site_state_matrix.npy'))
    radii = np.load(os.path.join(sites_dir, 'radii.npy'))

    rfilter = radii < rmax
    radii = radii[rfilter]
    S = S[:,rfilter]
    S /= np.linalg.norm(S,axis=0)

    for T in temps:
        print(T, end=' ')
        clust_cryst, cluster = cluster_crystallinity(perc_dir,cryst_mask,S,T,run_name,renormalise_by_cryst_size=False,return_cluster=True)
        clust_radii = radii[cluster]
        out = np.vstack((clust_radii, clust_cryst)).T
        save_npy(out, f'csc_v_radii-{n}-{T}K.npy',outdir)
    print('\n')






