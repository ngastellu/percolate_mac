#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours




ensembles = ['40x40', 'tempdot6', 'tempdot5']
rmaxs = [18.03, 121.2, 198.69]
official_labels = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
clrs = MAC_ensemble_colours()
temps = np.arange(180,440,10)
T = 300
zorders = [3,1,2]

motype = 'virtual'


setup_tex(fontsize=25)
fig, ax = plt.subplots()

for st, rmax, lbl, c, zz in zip(ensembles, rmaxs, official_labels, clrs,zorders):
    datadir = f'/Users/nico/Desktop/simulation_outputs/percolation/{st}/electronic_crystallinity/condsites_crysts_v_radii/rmax_{rmax}_sites_gammas_{motype}'
    nn = np.sort([int(dd.split('-')[1]) for dd in os.listdir(datadir)])
    cryst_fracs = np.load(f'/Users/nico/Desktop/simulation_outputs/structural_characteristics_MAC/fraction_cryst_atoms/frac_cryst_atoms_{st}.npy')
    if st == '40x40':
        cf = cryst_fracs[nn[0]-1]
    else:
        cf = cryst_fracs[nn[0]]
    radii, condsites_crysts = np.load(os.path.join(datadir, f'csc_v_radii-{nn[0]}-{T}K.npy')).T
    ax.scatter(condsites_crysts, radii, s=5.0,c=c,alpha=0.4,label=lbl,zorder=zz)
    for n in nn[1:]:
        radii, condsites_crysts = np.load(os.path.join(datadir, f'csc_v_radii-{n}-{T}K.npy')).T
        if st == '40x40':
            cf = cryst_fracs[n-1]
        else:
            cf = cryst_fracs[n]
        ax.scatter(condsites_crysts, radii,  s=5.0,c=c,alpha=0.4,zorder=zz)
    
ax.set_ylabel('Conducting site radius [\AA]')
ax.set_xlabel('Rescaled conducting site crystallinity')
ax.legend()

if motype == 'lo':
    ax.set_title(f'Lowest-energy MOs;  $T = {T}$K')
elif motype == 'virtual':
    ax.set_title(f'Mid-band virtual MOs;  $T = {T}$K')
else:
    ax.set_title(f'Highest-energy MOs;  $T = {T}$K')

plt.show()

