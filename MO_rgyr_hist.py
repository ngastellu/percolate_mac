#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours    
from qcnico.qchemMAC import MO_rgyr
from qcnico.coords_io import read_xyz
from qcnico.data_utils import save_npy




def get_lbls(Mdir):
    Mfiles = glob(os.path.join(Mdir, 'MOs_ARPACK*.npy'))
    nn = np.array([int(os.path.basename(f).split('-')[-1].split('.')[0]) for f in Mfiles])
    return nn




structypes = ['40x40', 'tempdot6', 'tempdot5']
synth_temps = ['500', 'q400', '300']
# lbls= ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']
lbls = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
motype = 'lo'

rmaxs = [18.03, 121.2, 198.69]
# temps = np.arange(180,440,10)

clrs = MAC_ensemble_colours()


setup_tex()
fig, axs = plt.subplots(3,1,sharex=True)
nbins = 171
# bins = np.linspace(0,1,nbins)
bins = np.linspace(0,170,nbins)
centers = (bins[1:] + bins[:-1]) * 0.5
dx = centers[1] - centers[0]

for ax, st, T, c, lbl in zip(axs, structypes, synth_temps, clrs, lbls):
    total_counts = np.zeros(bins.shape[0]-1)
    rmax = 0
    rmax_istruc = -1
    Mdir = f'/Users/nico/Desktop/simulation_outputs/percolation/{st}/MOs_ARPACK/{motype}'
    nn = get_lbls(Mdir)
    for n in nn:
        pos = read_xyz(f'/Users/nico/Desktop/scripts/disorder_analysis_MAC/structures/sAMC-{T}/sAMC{T}-{n}.xyz')
        M = np.load(os.path.join(Mdir, f'MOs_ARPACK_{motype}_{st}-{n}.npy'))
        radii = MO_rgyr(pos, M)
        rm = np.max(radii)
        if rm > rmax:
            rmax = rm
            rmax_istruc = n
            rmax_iMO = np.argmax(radii)
        counts, _ = np.histogram(radii, bins)
        total_counts += counts

    print(f'Max {motype} MO rgyr in {st} = {rmax} angstroms (MO {rmax_iMO} of sample {rmax_istruc}).')

    ax.bar(centers,total_counts, align='center', width=dx, color=c, label=lbl)
    ax.set_ylabel('Counts (log)')
    ax.set_yscale('log')
    ax.legend()

ax.set_xlabel('MO radii of gyration [\AA]')
plt.suptitle('Radii of 100 lowest-lying MOs')
plt.show()


