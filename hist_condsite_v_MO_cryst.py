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
fig, axs = plt.subplots(2,1,sharex=False)
nbins = 200
bins = np.linspace(0,1,nbins)
centers = (bins[1:] + bins[:-1]) / 2
dx = centers[1] - centers[0]

for st, rmax, lbl, c in zip(ensembles, rmaxs, official_labels, clrs):
    datadir = f'/Users/nico/Desktop/simulation_outputs/percolation/{st}/electronic_crystallinity/condsites_v_MOs_crysts/condsites_v_MOs_crystallinities_rmax_{rmax}_sites_gammas_{motype}'
    nn = [int(dd.split('-')[1]) for dd in os.listdir(datadir)]
    MO_crysts, condsites_crysts = np.load(os.path.join(datadir, f'sample-{nn[0]}', f'csc_v_Mc-{T}K.npy'))
    Mhist_total, _ = np.histogram(MO_crysts,bins)
    cs_hist_total, _ = np.histogram(condsites_crysts,bins)
    for n in nn[1:]:
        MO_crysts, condsites_crysts = np.load(os.path.join(datadir, f'sample-{n}', f'csc_v_Mc-{T}K.npy'))
        Mhist_sample, _ = np.histogram(MO_crysts, bins)
        cs_hist_sample, _ = np.histogram(condsites_crysts, bins)

        Mhist_total += Mhist_sample
        cs_hist_total += cs_hist_sample
    
    # Mhist_total = Mhist_total.astype('float') / Mhist_total.sum()
    # cs_hist_total = cs_hist_total.astype('float') / cs_hist_total.sum()
    axs[0].bar(centers, Mhist_total, align='center',width=dx,color=c,label=lbl,alpha=0.7)
    axs[1].bar(centers, cs_hist_total, align='center',width=dx,color=c,label=lbl,alpha=0.7)
    
axs[0].set_xlabel('MO crystallinity')
axs[1].set_xlabel('Conducting site crystallinity')

for ax in axs:
    ax.set_yscale('log')
    ax.set_ylabel('Counts (log)')
plt.tight_layout()
# plt.legend()

if motype == 'lo':
    plt.suptitle(f'Lowest-energy MOs;  $T = {T}$K')
elif motype == 'virtual':
    plt.suptitle(f'Mid-band virtual MOs;  $T = {T}$K')
else:
    plt.suptitle(f'Highest-energy MOs;  $T = {T}$K')

plt.show()

