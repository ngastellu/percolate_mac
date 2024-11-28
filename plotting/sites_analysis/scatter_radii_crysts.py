#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import MAC_ensemble_colours, setup_tex


structypes = ['40x40', 'tempdot6', 'tempdot5']
labels = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
clrs = MAC_ensemble_colours()
motype = 'virtual_w_HOMO'

setup_tex()

for st, lbl, c in zip(structypes,labels, clrs): 
    fig, ax = plt.subplots()
    radii = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/{st}/var_radii_data/all_site_radii_{motype}.npy')
    crysts = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/{st}/electronic_crystallinity/all_site_crysts_{motype}.npy')
    ax.scatter(radii,crysts,c=c,s=1.0,label=lbl)
    ax.set_xlim([0,220])
    ax.set_ylim([0,1])

    ax.set_xlabel('Site radius $a$ [\AA]')
    ax.set_ylabel('Site crystallinity $\chi$')
    plt.legend()
    plt.show()
