#!/usr/bin/env python


import matplotlib.pyplot as plt
from percplotting import dcrit_hists
from utils_analperc import get_dcrits


percdir = '/Users/nico/Desktop/simulation_outputs/percolation/tempdot6/percolate_output/zero_field/virt_100x100_gridMOs_rmax_121.2/'

with open(percdir + f'good_runs_rmax_121.2.txt') as fo:
    tdot6_lbls = [int(l.strip()) for l in fo.readlines()]


temps = [90,300]

dcrits = get_dcrits(tdot6_lbls, temps, percdir, pkl_prefix='out_percolate_rmax_121.2')

fig,ax= dcrit_hists(dcrits, temps, nbins=20, fontsize=40,alpha=1.0,show=False)
ax.set_xticks([10,15,20,25],fontsize=40)
plt.legend(fontsize=40)
plt.show()
