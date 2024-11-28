#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours



def gen_radii(ddir,eps,return_isample=False):
    sites_ddir = f'var_radii_data/to_local_sites_data/' 
    sampdirs = os.listdir(ddir + sites_ddir)
    print(sampdirs)
    # print(sampdirs)
    for d in sampdirs:
        # print(d)
        # print('\n'+d.split('-')[1], end=' ')
        try:
            # yield np.load(ddir + sites_ddir + d + f'/sites_data_{eps}/radii.npy')
            if return_isample:
                yield np.load(ddir + sites_ddir + d + '/sites_data_0.00105/radii.npy'), int(d.split('-')[-1])
            else:
                yield np.load(ddir + sites_ddir + d + '/sites_data_0.00105/radii.npy')
        except FileNotFoundError as e:
            print('NPY not found!')
            if return_isample:
                yield np.array([0]), int(d.split('-')[-1 ])
            else:
                yield np.array([0])


def gen_data(ddir,eps):
    sites_ddir = 'var_radii_data/to_local_sites_data/' 
    sampdirs = os.listdir(ddir + sites_ddir)
    # print(sampdirs)
    for d in sampdirs:
        # print('\n'+d.split('-')[1], end=' ')
        try:
            # yield np.load(ddir + sites_ddir + d + f'/sites_data_{eps}/radii.npy')
            yield np.load(ddir + sites_ddir + d + f'/sites_data_{eps}/rr_v_masses_v_iprs_v_ee.npy')[2,:]
        except FileNotFoundError as e:
            # print('NPY not found!')
            yield np.array([0])

simdir = '/Users/nico/Desktop/simulation_outputs/'
percdir = simdir + 'percolation/'

motype = 'virtual_w_HOMO'

structypes = ['40x40', 'tempdot6', 'tempdot5']
lbls= ['sAMC-500', 'sAMC-q400', 'sAMC-300']
clrs = MAC_ensemble_colours()

bins = np.arange(200)
centers = (bins[:-1] + bins[1:]) * 0.5
dx = bins[1] - bins[0]

setup_tex()
fig, axs = plt.subplots(3,1,sharex=True)

for ax, st, lbl, c in zip(axs,structypes,lbls,clrs):
    ntiny = 0
    datadir = percdir + st + '/'
    radii = np.load(os.path.join(percdir, st, f'var_radii_data/all_site_radii_{motype}.npy'))
    rmax = np.max(radii)
    hist = np.histogram(radii,bins=bins)[0]
    print(f'****** MAX RADII INFO FOR {st} ******')
    print('\trmax = ', rmax)
    ax.bar(centers, hist,align='center',width=dx,color=c,label=lbl)
    ax.legend()
    ax.set_yscale('log')

plt.show()