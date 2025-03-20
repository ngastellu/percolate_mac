#!/usr/bon/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from qcnico.plt_utils import setup_tex



ensemble = '40x40'
motypes = ['lo', 'virtual_w_HOMO', 'hi']
datadir = f'/Users/nico/Desktop/simulation_outputs/percolation/{ensemble}/MO_ipr_v_MO_cryst'
npys = os.listdir(os.path.join(datadir, 'virtual_w_HOMO')) # could be any MO type, will yield same results

setup_tex(fontsize=30)
fig, ax = plt.subplots()

for npy in npys:
    nstruc = npy.split('.')[0]
    all_data = [np.load(os.path.join(datadir, mt, npy)) for mt in motypes]
    energies = [dat[0,:] for dat in all_data]
    iprs = np.hstack([dat[1,:] for dat in all_data])
    crysts = np.hstack([dat[2,:] for dat in all_data])
    eF = (energies[1][0] + energies[1][1]) * 0.5

    energies = np.hstack(energies) - eF

    plt_out = ax.scatter(iprs,crysts,c=np.abs(energies),alpha=0.8,s=3.0)

fig.colorbar(plt_out, ax=ax)
ax.set_xlabel('IPR')
ax.set_ylabel('MO crystallinity $\chi$')
plt.show()