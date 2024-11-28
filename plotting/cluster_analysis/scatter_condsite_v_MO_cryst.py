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

motype = 'lo'


setup_tex(fontsize=25)
fig, ax = plt.subplots()

for st, rmax, lbl, c in zip(ensembles, rmaxs, official_labels, clrs):
    datadir = f'/Users/nico/Desktop/simulation_outputs/percolation/{st}/electronic_crystallinity/condsites_v_MOs_crysts/condsites_v_MOs_crystallinities_rmax_{rmax}_sites_gammas_{motype}'
    nn = [int(dd.split('-')[1]) for dd in os.listdir(datadir)]
    MO_crysts, condsites_crysts = np.load(os.path.join(datadir, f'sample-{nn[0]}', f'csc_v_Mc-{T}K.npy'))
    ax.scatter(MO_crysts, condsites_crysts, s=3.0,c=c,alpha=0.5,label=lbl)
    for n in nn[1:]:
        MO_crysts, condsites_crysts = np.load(os.path.join(datadir, f'sample-{n}', f'csc_v_Mc-{T}K.npy'))
        ax.scatter(MO_crysts, condsites_crysts, s=3.0,c=c,alpha=0.5)
    
ax.plot(np.linspace(0,1,100),np.linspace(0,1,100),'k--',lw=0.9)
ax.set_xlabel('MO crystallinity')
ax.set_ylabel('Conducting site crystallinity')
ax.legend()

if motype == 'lo':
    ax.set_title(f'Lowest-energy MOs;  $T = {T}$K')
elif motype == 'virtual':
    ax.set_title(f'Mid-band virtual MOs;  $T = {T}$K')
else:
    ax.set_title(f'Highest-energy MOs;  $T = {T}$K')

plt.show()

