#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours



simdir = '/Users/nico/Desktop/simulation_outputs/'
percdir = simdir + 'percolation/'

structypes = ['40x40', 'tempdot6', 'tempdot5']
lbls= ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']
clrs = MAC_ensemble_colours()


setup_tex()
fig, axs = plt.subplots(3,1,sharex=True)

for ax, st, lbl, c in zip(axs,structypes,lbls,clrs):
    centers, hist = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/{st}/site_energy_hists/energy_histogram_virtual.npy').T
    dx = centers[1] - centers[0]
    ax.bar(centers, hist,align='center',width=dx,color=c,label=lbl)
    ax.legend()
    # ax.set_yscale('log')
    # print(f'{st} ensemble has {ntiny} radii <= 1')

plt.show()