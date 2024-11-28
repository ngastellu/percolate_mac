#!/usr/bin/env python

import numpy as np
import os
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours


structypes = ['40x40', 'tempdot6', 'tempdot5']
lbls = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
clrs = MAC_ensemble_colours()

percdir = '/Users/nico/Desktop/simulation_outputs/percolation/'

setup_tex()
fig, axs = plt.subplots(3,1,sharex=True)

bins = np.linspace(0,82,201)

for st, lbl, ax, c in zip(structypes,lbls,axs,clrs):
    rdir = percdir + f'{st}/var_radii_data/radii_0.00105_psi_pow2_hi/'
    radii = np.hstack([np.load(rdir + f) for f in os.listdir(rdir)]) 
    print('MAX RADIUS = ', np.max(radii))
    hist, bin_edges = np.histogram(radii, bins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2 
    
    dx = centers[1] - centers[0]

    ax.bar(centers, hist,align='center',width=dx,color=c,label=lbl)
    ax.legend()
    ax.set_ylabel('Counts (log)')
    ax.set_yscale('log')
    # print(f'{st} ensemble has {ntiny} radii <= 1')

ax.set_xlabel('Site radius $a$ [\AA]')# / \# crystalline atoms in structure')

plt.suptitle('Radii of sites from 100 highest-energy MOs')

plt.show()

