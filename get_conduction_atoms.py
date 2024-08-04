#!/usr/bin/env python

import os
import sys
import numpy as np
import pickle
from var_a_percolate_mac.percolate import diff_arrs_var_a
from qcnico.coords_io import read_xyz
from scipy.spatial import KDTree


nn = int(sys.argv[1])
rmax = float(sys.argv[2])

rho = 1.05e-3
kB = 8.617e-5

outdir=f'sample-{nn}/conduction_atoms'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

sitesdir = f'sample-{nn}/new_sites_data_{rho}/'

cc = np.load(sitesdir + 'centers.npy')
ee = np.load(sitesdir + 'ee.npy')
ii = np.load(sitesdir + 'ii.npy')
radii = np.load(sitesdir + 'radii.npy')
S = np.load(sitesdir + 'site_state_matrix.npy')

igood = (radii < rmax).nonzero()[0]
centres = cc[igood]
ee = ee[igood]
ii = ii[igood]
radii = radii[igood]
S = S[:,igood]
N = S.shape[0]

temps = np.arange(40,440,10)
edarr, rdarr, _ = diff_arrs_var_a(ee,cc,radii,eF=0,E=np.array([0.0,0.0]),include_r_prefactor=True)


for T in temps:
    print(f'T = {T}K')
    pklfile = f'sample-{nn}/out_percolate_rmax_{rmax}-{T}K.pkl'
    with open(pklfile,'rb') as fo:
        data = pickle.load(fo)
        clusters = data[0]
        dcrit = data[1]

    if len(clusters) == 1:
        perc_cluster = clusters[0]
    else:
        dists = rdarr + (edarr/(kB*T))
        ij = (dists == dcrit).nonzero()
        print(ij)
        for k, c in enumerate(clusters):
            if i in c and j in c:
                perc_cluster = c
                break
    
    atoms_in_cluster = np.zeros(N,dtype=bool)
    
    for n in perc_cluster:
        site_ket = S[:,n]
        rel_atoms = site_ket.nonzero()[0] # C atoms on which |s_n> has nonzero amplitude 
        atoms_in_cluster[rel_atoms] = True
    
    np.save(f'{outdir}/cluster_atoms_indices-{nn}-{T}K.npy', atoms_in_cluster.nonzero()[0])
