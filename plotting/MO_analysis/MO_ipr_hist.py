#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours




structypes = ['40x40', 'tempdot6', 'tempdot5']
official_labels = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
clrs = MAC_ensemble_colours()
temps = np.arange(180,440,10)
T = 300
nbins = 100

motypes = ['lo','virtual_w_HOMO','hi']

setup_tex(fontsize=32)
fig, axs = plt.subplots(3,1,sharex=True,sharey=True)
nbins = 200


for ax, mt, in zip(axs,motypes):
    for st, c, lbl in zip(structypes,clrs,official_labels):
        datadir = f'/Users/nico/Desktop/simulation_outputs/percolation/{st}/MO_ipr_v_MO_cryst/{mt}'
        npys = os.listdir(datadir)
        data = np.hstack([np.load(os.path.join(datadir, f))[0,:] for f in npys])
        hist, bins = np.histogram(data,nbins)
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
    ax.set_yscale('log')

    # ax.axvline(x=centers[np.argmax(hist_renormd)],ymin=0,ymax=1,c=c,ls='--',lw=0.8)
    # ax.set_yscale('log')
    # print(f'{st} ensemble has {ntiny} radii <= 1')

    # print(f'Bounds for {st} = ', [np.min(data),np.max(data)])
    # print((data > 1).sum() / data.shape[0])

ax.set_xlabel('IPR')

axs[0].legend(fontsize=25)

# plt.suptitle('100 lowest- and highest-lying MOs')

plt.show()


