#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours
import os
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple



structypes = ['40x40', 'tempdot6', 'tempdot5']
#lbls= ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']
lbls= ['sAMC-500', 'sAMC-q400', 'sAMC-300']
clrs = MAC_ensemble_colours()

renormd = False

simdir = '/Users/nico/Desktop/simulation_outputs/'
percdir = simdir + 'percolation/'

setup_tex(fontsize=25)
fig, axs = plt.subplots(3,1,sharex=True)
nbins = 200
if renormd:
    bins = np.linspace(0,1.8,nbins)
else:
    bins = np.linspace(0,1,nbins)


motypes = ['lo','hi','virtual']

# rcParams['figure.figsize'] = [14,12]

for ax, mt, in zip(axs,motypes):
    for st, c, lbl in zip(structypes,clrs,lbls):
        if renormd:
            datadir_renormd = percdir + f'{st}/electronic_crystallinity/MO_crystallinities_frac_renormd/{mt}_MO_crystallinities_crystalline_{st}_frac_renormd/'
            npys_renormd = [datadir_renormd + f for f in os.listdir(datadir_renormd)]
            data = np.hstack([np.load(f) for f in npys_renormd])
        else:
            datadir = percdir + f'{st}/electronic_crystallinity/MO_crystallinities_unnormd/{mt}_MO_crystallinities_crystalline_{st}/'
            npys = [datadir + f for f in os.listdir(datadir)]
            data = np.hstack([np.load(f) for f in npys])

        hist, _ = np.histogram(data,bins)
        centers = (bins[1:] + bins[:-1]) * 0.5
        dx = centers[1] - centers[0]

        h1 = ax.bar(centers, hist,align='center',width=dx,color=c,zorder=1,alpha=0.7,label=lbl)


    # datadir_hi_renormd = percdir + f'{st}/electronic_crystallinity/MO_crystallinities_frac_renormd/hiMO_crystallinities_crystalline_{st}_frac_renormd/'
    # npys_hi_renormd = [datadir_hi_renormd + f for f in os.listdir(datadir_hi_renormd)]
    # data_hi_renormd = np.hstack([np.load(f) for f in npys_hi_renormd])

    # datadir_virt_renormd = percdir + f'{st}/electronic_crystallinity/MO_crystallinities_frac_renormd/MO_crystallinities_crystalline_{st}_frac_renormd/'
    # npys_virt_renormd = [datadir_virt_renormd + f for f in os.listdir(datadir_virt_renormd)]
    # data_virt_renormd = np.hstack([np.load(f) for f in npys_virt_renormd])

    
    


    # data = np.hstack((data_lo,data_hi))

    # h1 = ax.bar(centers, hist,align='center',width=dx,color=c,zorder=1)
    # h2 = ax.bar(centers, hist_renormd,align='center',width=dx,color=c,alpha=0.5,zorder=2)
    # l = ax.legend([(h1,h2)], [lbl], handler_map={tuple: HandlerTuple(ndivide=None)})
    ax.set_ylabel('Counts')

    # ax.axvline(x=centers[np.argmax(hist_renormd)],ymin=0,ymax=1,c=c,ls='--',lw=0.8)
    # ax.set_yscale('log')
    # print(f'{st} ensemble has {ntiny} radii <= 1')

    # print(f'Bounds for {st} = ', [np.min(data),np.max(data)])
    # print((data > 1).sum() / data.shape[0])

ax.set_xlabel('MO crystallinity $\chi$')
ax.legend()

# plt.suptitle('100 lowest- and highest-lying MOs')

plt.show()


