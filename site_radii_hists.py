#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours



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

eps = '0.00105'

structypes = ['40x40', 'tempdot6', 'tempdot5']
lbls= ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']
clrs = MAC_ensemble_colours()

# bins = np.arange(302)
bins = np.linspace(0,11,100)
centers = (bins[:-1] + bins[1:]) * 0.5
dx = bins[1] - bins[0]

setup_tex()
fig, axs = plt.subplots(3,1,sharex=True)

for ax, st, lbl, c in zip(axs,structypes,lbls,clrs):
    nzeros = 0
    datadir = percdir + st + '/'
    radii_gen = gen_data(datadir,eps)
    radii0 = next(radii_gen)
    nzeros += (radii0 <= 1).sum()
    hist, _ = np.histogram(radii0,bins=bins)
    for radii in radii_gen:
        hist += np.histogram(radii,bins=bins)[0]
        nzeros += (radii <= 1).sum()
    ax.bar(centers, hist,align='center',width=dx,color=c,label=lbl)
    ax.legend()
    ax.set_yscale('log')
    print(f'{st} ensemble has {nzeros} radii <= 1')

plt.show()