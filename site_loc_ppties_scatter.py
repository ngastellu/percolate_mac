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
            data = np.load(ddir + sites_ddir + d + f'/sites_data_{eps}/rr_v_masses_v_iprs_v_ee.npy')
            yield data[:2,:]
        except FileNotFoundError as e:
            # print('NPY not found!')
            yield np.array([0,0])

simdir = '/Users/nico/Desktop/simulation_outputs/'
percdir = simdir + 'percolation/'

eps = '0.003'

structypes = ['40x40', 'tempdot6', 'tempdot5']
lbls= ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']
clrs = MAC_ensemble_colours()

bins = np.arange(302)
centers = (bins[:-1] + bins[1:]) * 0.5
dx = bins[1] - bins[0]

setup_tex()
fig, ax = plt.subplots()

for st, lbl, c in zip(structypes,lbls,clrs):
    datadir = percdir + st + '/'
    rm_gen = gen_data(datadir,eps)
    radii, masses = next(rm_gen)
    ax.scatter(radii,masses,c=c,alpha=0.4,label=lbl,s=5.0)
    for rm in rm_gen:
        radii, masses = rm
        ax.scatter(radii,masses,c=c,alpha=0.4,s=5.0)

ax.set_xlabel('Site radii [\AA]')
ax.set_ylabel('Site mass')
plt.legend()
plt.show()