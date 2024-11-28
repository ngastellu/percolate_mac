#!/usr/bin/env python

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from qcnico.qcplots import plot_MO
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons



rCC = 1.8

simdir = '/Users/nico/Desktop/simulation_outputs/percolation/'
run_type = 'virt_100x100_gridMOs_eps_rho_1.05e-3'
npyname = 'rr_v_masses_v_ee_eps_rho_0.00105.npy' 

outer_posdir = '/Users/nico/Desktop/simulation_outputs/MAC_structures/'

inner_posdirs= ['pCNN/bigMAC_40x40_ensemble/', 'Ata_structures/relaxed_structures/tempdot6/', 'Ata_structures/relaxed_structures/tempdot5/']
outer_dirs = [simdir + d for d in ['40x40/', 'tempdot6/', 'tempdot5/']]
posprefix = ['bigMAC-', 'tempdot6n', 'tempdot5n']
rmaxs = [18.03, 121.2, 198.69]

dd_rings = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'

lbls = ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']


# for k, d in enumerate(outer_dirs):
rmax = rmaxs[-1]
run_type = f'virt_100x100_gridMOs_rmax_{rmax}'
d = outer_dirs[-1]
k = 2
print(d)
percdir = d + f'percolate_output/zero_field/{run_type}/'
good_runs_file = percdir + f'good_runs_rmax_{rmax}.txt'
Mdir = d + 'MOs_ARPACK/virtual/'
posdir = outer_posdir + inner_posdirs[k]

fo = open(good_runs_file)
lines = fo.readlines()
fo.close()
nn = set([int(l.strip()) for l in lines])

nM = set([int(f.split('-')[1].split('.')[0]) for f in glob(Mdir + '*npy')])

nn = np.array(list(nn & nM))

istrucs = np.random.choice(nn,size=3,replace=False)

for i in istrucs:
    pos, _ = read_xsf(posdir + f'{posprefix[k]}{i}_relaxed.xsf')
    pos = remove_dangling_carbons(pos,rCC)
    
    sites_dir = d + f'var_radii_data/to_local_sites_data/sample-{i}/sites_data_0.00105/'
    all_radii = np.load(sites_dir + 'radii.npy')
    all_ii = np.load(sites_dir + 'ii.npy')
    all_centers = np.load(sites_dir + 'centers.npy')

    M = np.load(Mdir + f'MOs_ARPACK_bigMAC-{i}.npy')

    isorted = np.argsort(all_radii)
    nsites = isorted.shape[0]

    # only sample the top 10% radii (in size)
    isample_range = np.arange(int(0.9*nsites),nsites-1) 
    isample = np.random.choice(isample_range,size=3,replace=False)
    isample = np.hstack([isample, [nsites-1]]) #always include max radius

    iisampled = np.unique(all_ii[isample])

    for n in iisampled:
        jj = (all_ii == n).nonzero()[0]
        rr = all_radii[jj]
        cc = all_centers[jj]

        clrs = ['r'] * cc.shape[0]

        l = np.argmax(rr)
        clrs[l] = 'limegreen'

        plot_MO(pos, M, n,dotsize=0.5,loc_centers=cc,loc_radii=rr,c_clrs=clrs)









