#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from utils_analperc import arrhenius_fit, get_dcrits, saddle_pt_sigma
from qcnico.plt_utils import setup_tex



# Step 1: Get proportion of crystalline hexagons
dd_rings = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'

ring_data_tdot25 = np.load(dd_rings + 'avg_ring_counts_tdot25.npy')
ring_data_t1 = np.load(dd_rings + 'avg_ring_counts_t1.npy')
ring_data_pCNN = np.load(dd_rings + 'avg_ring_counts_normalised.npy')

p6c_tdot25 = ring_data_tdot25[3] / ring_data_tdot25.sum()
p6c_t1 = ring_data_t1[3] / ring_data_t1.sum()
p6c_pCNN = ring_data_pCNN[3]

p6c = np.array([p6c_tdot25, p6c_pCNN,p6c_t1])


# Step 2: Get sigma(T) for each percolation run
temps = np.arange(40,440,10)

dd_tdot25 = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tdot25/percolate_output/zero_field/extremal_MOs/'
dd_t1 = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/t1/percolate_output/zero_field/extremal_MOs/'
dd_pCNN = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/extremal_MOs/'

with open(dd_pCNN + 'lo50/good_runs.txt') as fo: # {good_runs(lo50)} is a subset of {good_runs(hi50)}
    pCNN_lbls = [int(l.strip()) for l in fo.readlines()]

mo_types = ['lo50', 'hi50']


setup_tex()
fig, ax = plt.subplots()

for mt in mo_types:
    sigmas = np.zeros(3)
    ddtdot25 = dd_tdot25 +  f'{mt}/'
    ddt1 = dd_t1 + f'{mt}/'
    ddpCNN = dd_pCNN +  f'{mt}/'

    dcrits_tdot25 = get_dcrits(range(7),temps,ddtdot25)
    dcrits_t1 = get_dcrits(range(7),temps,ddt1)
    dcrits_pCNN = get_dcrits(pCNN_lbls, temps, ddpCNN)

    sigmas = [saddle_pt_sigma(dcrits) for dcrits in [dcrits_tdot25,dcrits_pCNN,dcrits_t1]]
    x0 = [15,15,0]
    eas = np.zeros(3)
    for k in range(3):
        eas[k], _  = arrhenius_fit(temps, sigmas[k], inv_T_scale=1000.0,x_start=x0[k])
    ax.plot(p6c, eas*1000, label=mt[:2])
    

ax.set_xlabel('$p_{6c}$')
ax.set_ylabel('$E_a$ [meV]')

plt.legend()
plt.show()
    






