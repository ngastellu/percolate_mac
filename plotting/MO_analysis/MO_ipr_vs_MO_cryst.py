#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours




ensembles = ['40x40', 'tempdot6', 'tempdot5']
official_labels = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
clrs = MAC_ensemble_colours()

motype = 'lo'


setup_tex(fontsize=25)
fig, ax = plt.subplots()

for st, lbl, c in zip(ensembles, official_labels, clrs):
    datadir = f'/users/nico/desktop/simulation_outputs/percolation/{st}/mo_ipr_v_mo_cryst/{motype}'
    npys = os.listdir(datadir)
    dat = np.load(os.path.join(datadir, npys[0]))
    print(dat.shape)
    iprs, MO_crysts = dat
    ax.scatter(iprs, MO_crysts, s=3.0,c=c,alpha=0.5,label=lbl)
    for npy in npys[1:]:
        dat = np.load(os.path.join(datadir, npy))
        iprs, MO_crysts = dat
        ax.scatter(iprs, MO_crysts, s=3.0,c=c,alpha=0.5)
    
# ax.plot(np.linspace(0,1,100),np.linspace(0,1,100),'k--',lw=0.9)
ax.set_xlabel('IPR')
ax.set_ylabel('MO crystallinity $\chi$')
ax.legend()

if motype == 'lo':
    ax.set_title(f'Lowest-energy MOs;  $T = {T}$K')
elif motype == 'virtual_w_HOMO':
    ax.set_title(f'Mid-band virtual MOs;  $T = {T}$K')
else:
    ax.set_title(f'Highest-energy MOs;  $T = {T}$K')

plt.show()

