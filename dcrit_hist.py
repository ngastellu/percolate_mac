#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours, multiple_histograms



simdir = '/Users/nico/Desktop/simulation_outputs/'
percdir = simdir + 'percolation/'

structypes = ['40x40', 'tempdot6', 'tempdot5']
lbls= ['sAMC-500', 'sAMC-q400', 'sAMC-300']
rmaxs = [18.03, 121.2, 198.69]
clrs = MAC_ensemble_colours()

# motypes = ['kBTlo', 'virtual', 'kBThi']

setup_tex()
# fig, axs = plt.subplots(3,1,sharex=True)
# for ax, mt in zip(axs,motypes):
#     vals_arr = []
#     for st, rmax in zip(structypes, rmaxs):
#         vals_arr.append(np.load(percdir + f'{st}/dcrits/dcrits_rmax_{rmax}_300K_{mt}.npy'))
#     fig, ax = multiple_histograms(vals_arr,lbls,colors=clrs,show=False, plt_objs=(fig,ax),alpha=0.7)

#     ax.set_yscale('log')
#     # print(f'{st} ensemble has {ntiny} radii <= 1')

# axs[0].legend()
# axs[2].set_xlabel('Critical distance $\\xi_c$')

vals_arr = []
for st, rmax in zip(structypes, rmaxs):
    vals_arr.append(np.load(percdir + f'{st}/dcrits/dcrits_rmax_{rmax}_300K_kBTlo.npy'))
fig, ax = multiple_histograms(vals_arr,lbls,colors=clrs,show=False,alpha=0.7)

ax.set_yscale('log')
ax.set_xlabel('Critical distance $\\xi_c$')

ax.set_title('Distribution of percolation thresholds for $\mu=\epsilon_0$ calculations.')
plt.legend()

plt.show()