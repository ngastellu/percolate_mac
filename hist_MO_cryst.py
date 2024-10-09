#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours
import os


structypes = ['40x40', 'tempdot6', 'tempdot5']
#lbls= ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']
lbls= ['sAMC-500', 'sAMC-q400', 'sAMC-300']
clrs = MAC_ensemble_colours()

simdir = '/Users/nico/Desktop/simulation_outputs/'
percdir = simdir + 'percolation/'

setup_tex(fontsize=25)
fig, axs = plt.subplots(3,1,sharex=True)
nbins = 200
bins = np.linspace(0,6.6,nbins)

# rcParams['figure.figsize'] = [14,12]

for ax, st, lbl, c in zip(axs,structypes,lbls,clrs):

    # datadir_lo = percdir + f'{st}/electronic_crystallinity/MO_crystallinities_unnormd/loMO_crystallinities_crystalline_{st}/'
    # npys_lo = [datadir_lo + f for f in os.listdir(datadir_lo)]
    # data_lo = np.hstack([np.load(f) for f in npys_lo])

    # datadir_hi = percdir + f'{st}/electronic_crystallinity/MO_crystallinities_unnormd/hiMO_crystallinities_crystalline_{st}/'
    # npys_hi = [datadir_hi + f for f in os.listdir(datadir_hi)]
    # data_hi = np.hstack([np.load(f) for f in npys_hi])

    # datadir_virt = percdir + f'{st}/electronic_crystallinity/MO_crystallinities_unnormd/MO_crystallinities_crystalline_{st}/'
    # npys_virt = [datadir_virt + f for f in os.listdir(datadir_virt)]
    # data_virt = np.hstack([np.load(f) for f in npys_virt])

    datadir_lo_renormd = percdir + f'{st}/electronic_crystallinity/MO_crystallinities_frac_renormd/loMO_crystallinities_crystalline_{st}_frac_renormd/'
    npys_lo_renormd = [datadir_lo_renormd + f for f in os.listdir(datadir_lo_renormd)]
    data_lo_renormd = np.hstack([np.load(f) for f in npys_lo_renormd])

    # datadir_hi_renormd = percdir + f'{st}/electronic_crystallinity/MO_crystallinities_frac_renormd/hiMO_crystallinities_crystalline_{st}_frac_renormd/'
    # npys_hi_renormd = [datadir_hi_renormd + f for f in os.listdir(datadir_hi_renormd)]
    # data_hi_renormd = np.hstack([np.load(f) for f in npys_hi_renormd])

    # datadir_virt_renormd = percdir + f'{st}/electronic_crystallinity/MO_crystallinities_frac_renormd/MO_crystallinities_crystalline_{st}_frac_renormd/'
    # npys_virt_renormd = [datadir_virt_renormd + f for f in os.listdir(datadir_virt_renormd)]
    # data_virt_renormd = np.hstack([np.load(f) for f in npys_virt_renormd])

    
    


    # data = np.hstack((data_lo,data_hi))
    data = data_lo_renormd
    hist, bins = np.histogram(data,bins=bins)
    centers = (bins[1:] + bins[:-1]) * 0.5
    dx = centers[1] - centers[0]

    ax.bar(centers, hist,align='center',width=dx,color=c,label=lbl)
    ax.legend()
    ax.set_ylabel('Counts')
    # ax.set_yscale('log')
    # print(f'{st} ensemble has {ntiny} radii <= 1')

    print(f'Bounds for {st} = ', [np.min(data),np.max(data)])

ax.set_xlabel('Normalised MO crystallinity $\chi/N_c$')

# plt.suptitle('100 lowest- and highest-lying MOs')

plt.show()


