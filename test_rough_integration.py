#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex
from utils_analperc import get_dcrits, rough_integral_sigma, saddle_pt_sigma
from percplotting import dcrit_hists


temps = np.arange(40,440,10)

pCNNdir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/zero_field/100x100_gridMOs/'

with open(pCNNdir + 'good_runs.txt') as fo:
    pCNN_lbls = [int(l.strip()) for l in fo.readlines()]


dcrits = get_dcrits(pCNN_lbls, temps, pCNNdir)

sigma_rough = rough_integral_sigma(dcrits, nbins=30)
sigma_saddle = saddle_pt_sigma(dcrits, nbins=30)

setup_tex()
fig, ax = plt.subplots()

ax.plot(1000/temps,np.log(sigma_rough),'ro-',label='new')
ax.plot(1000/temps,np.log(sigma_saddle),'bo-',label='old')
ax.set_xlabel('$1000/T$')
ax.set_ylabel('$\log\sigma$')
ax.set_title('Testing new method of estimating $\sigma(T)$')
plt.legend()
plt.show()

dcrit_hists(dcrits, temps, 30, plot_inds=[0,10,30,39])



