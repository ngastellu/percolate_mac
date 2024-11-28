#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours
import os


def get_struc_labels(ddir, prefix='sample',connector_char='-'):
    return np.array([int(x.split(connector_char)[-1]) for x in glob(f'{ddir}/{prefix}{connector_char}*')])

def concatenate_data(nn, temps, ddir, dir_prefix='sample-', npy_prefix='clust_cryst-'):
    full_data = np.hstack([np.load(f'{ddir}/{dir_prefix}{nn[0]}/{npy_prefix}{T}K.npy').flatten() for T in temps])   
    for n in nn[1:]:
        dats = np.hstack([np.load(f'{ddir}/{dir_prefix}{n}/{npy_prefix}{T}K.npy').flatten() for T in temps])
        full_data = np.hstack((full_data, dats))
    return full_data


if __name__ == '__main__':
        
            
    structypes = ['40x40', 'tempdot6', 'tempdot5']
    # lbls= ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']
    lbls = ['sAMC-500', 'sAMC-q400', 'sAMC-300']

    rmaxs = [18.03, 121.2, 198.69]
    # temps = np.arange(180,440,10)
    temps = [180]
    
    clrs = MAC_ensemble_colours()

    simdir = '/Users/nico/Desktop/simulation_outputs/'
    percdir = simdir + 'percolation/'

    setup_tex()
    fig, axs = plt.subplots(3,1,sharex=True)
    nbins = 100
    # bins = np.linspace(0,1,nbins)
    bins = np.linspace(0,1.2e-4,nbins)

    for ax, st, r, lbl, c in zip(axs,structypes, rmaxs, lbls,clrs):

        datadir = percdir + f'{st}/electronic_crystallinity/cluster_crystallinities_rmax_{r}/'
        nn = get_struc_labels(datadir)
        # data = np.hstack([[np.load(f'sample-{n}/clust_cryst-{T}K.npy') for T in temps] for n in nn])
        data = concatenate_data(nn, temps, datadir)
        print(f'Max site crystallinity in {st} = {np.max(data)}')
        hist, bins = np.histogram(data,bins=bins)
        centers = (bins[1:] + bins[:-1]) * 0.5
        dx = centers[1] - centers[0]

        ax.bar(centers, hist,align='center',width=dx,color=c,label=lbl)
        ax.legend()
        ax.set_ylabel('Counts')
        # ax.set_yscale('log')
        # print(f'{st} ensemble has {ntiny} radii <= 1')

    ax.set_xlabel('Conducting site crystallinity $\chi$')# / \# crystalline atoms in structure')

    # plt.suptitle('Adjusted for number of crystalline atoms in structure')

    plt.show()
