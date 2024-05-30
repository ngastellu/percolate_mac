#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
from qcnico import plt_utils


def gen_data(lbls, ddir, filename='rr_v_masses_v_ee.npy'):
    for n in lbls:
        # print(n)
        npy = ddir + f'sample-{n}/{filename}'
        yield np.load(npy)

def gen_n_sites(lbls,ddir):
    for n in lbls:
        sdir = ddir + f'sample-{n}/'
        radii = np.load(sdir + 'radii.npy')
        nradii = radii.shape[0]
        ii = np.load(sdir + 'ii.npy')
        iunique, counts = np.unique(ii,return_counts=True)
        out = np.zeros((nradii,2))
        for i in range(nradii):
            iMO = ii[i]
            j = (iunique == iMO).nonzero()[0]
            # print(j)
            nsites = counts[j]
            out[i,0] = radii[i]
            out[i,1] = nsites
        yield out



simdir = '/Users/nico/Desktop/simulation_outputs/percolation/'
run_type = 'virt_100x100_gridMOs_eps_rho_1.05e-3'
npyname = 'rr_v_masses_v_ee_eps_rho_0.00105.npy' 

outer_dirs = [simdir + d for d in ['40x40/', 'Ata_structures/tempdot6/', 'Ata_structures/tempdot5/']]

dd_rings = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'

ring_data_tempdot5 = np.load(dd_rings + 'avg_ring_counts_tempdot5_new_model_relaxed.npy')
ring_data_pCNN = np.load(dd_rings + 'avg_ring_counts_normalised.npy')
ring_data_tempdot6 = np.load(dd_rings + 'avg_ring_counts_tempdot6_new_model_relaxed.npy')

p6c_tempdot6 = ring_data_tempdot6[4] / ring_data_tempdot6.sum()
p6c_tempdot5 = ring_data_tempdot5[4] / ring_data_tempdot5.sum()
p6c_pCNN = ring_data_pCNN[4]
# p6c = np.array([p6c_tdot25, p6c_pCNN,p6c_t1,p6c_tempdot6])
p6c = np.array([p6c_pCNN,p6c_tempdot6,p6c_tempdot5])

clrs = plt_utils.get_cm(p6c, 'inferno',min_val=0.25,max_val=0.7)
lbls = ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']

plt_utils.setup_tex()

rmax = 30

istrucs = np.zeros((3,3),dtype=int) # structure inds of the high-mass sites

nsites = [104424, 77544, 56588]


for k, d in enumerate(outer_dirs):
    ddir  = d + 'var_radii_data/' + run_type + '/'
    percdir = d + f'percolate_output/zero_field/{run_type}/'
    good_runs_file = percdir + 'good_runs_eps_rho_1.05e-3.txt'
    nbig = 0

    fo = open(good_runs_file)
    lines = fo.readlines()
    fo.close()
    nn = [int(l.strip()) for l in lines]
    datgen = gen_data(nn, ddir, filename=npyname)
    
    fig, ax = plt.subplots() 

    for dat in datgen:
        # print(dat.shape)
        rr, masses, ee = dat
        densities = masses / (np.pi * rr * rr)
        ee -= np.min(ee)
        ax.scatter(rr, densities, c=clrs[k],s=1.0)
        # nsites += rr.shape[0]
        nbig += (rr>rmax).sum()

    ax.axvline(x=rmax,ymin=0,ymax=1,c='k')
    ax.set_xlabel('Site radius [\AA]')
    ax.set_ylabel('Site probability density')

    plt.show()

    print(f'Nb of sites with r > {rmax} angstroms = {nbig} / {nsites[k]}')
