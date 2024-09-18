#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from qcnico.plt_utils import MAC_ensemble_colours, setup_tex




def average_cluster_crystallinity(datadir, temps):
    """For each structure of a given ensemble (specified by `datadir`), returns the average crystallinity of all sites contained in 
    a conducting cluster.
    The average is performed over all sites in clusters from all temperatures"""
    nn = np.sort([int(d.split('-')[1]) for d in os.listdir(datadir)])
    avg_crysts = np.zeros(len(nn))
    for k, n in enumerate(nn):
        print(f' -- {n} --')
        npys = [datadir + f'sample-{n}/clust_cryst-{T}K.npy' for T in temps]
        avg_crysts[k] = np.mean(np.hstack([np.load(npy) for npy in npys]))
    
    return nn, avg_crysts
        



ensembles = ['40x40', 'tempdot6', 'tempdot5']
official_labels = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
max_radii = [18.03, 121.2, 198.69]
clrs = MAC_ensemble_colours()
temps = np.arange(180,440,10)

percdir = os.path.expanduser('~/Desktop/simulation_outputs/percolation/')
strucdir = os.path.expanduser('~/Desktop/simulation_outputs/structural_characteristics_MAC/fraction_cryst_atoms/') 


setup_tex()
fig, ax = plt.subplots()


for st, rmax, c, lbl in zip(ensembles,max_radii,clrs,official_labels):
    print(f'***** {st} *****')
    clustcryst_dir = percdir + f'{st}/electronic_crystallinity/cluster_crystallinities_rmax_{rmax}/'
    sitcryst_dir = percdir + f'{st}/electronic_crystallinity/sites_crystallinities_crystalline_{st}/all_sites/'
    nn_c, avg_clustcryst = average_cluster_crystallinity(clustcryst_dir, temps)

 
    # if st == '40x40':
    #     nn_c -= 1 # indices in 40x40 ensemble start at 1

    igood = avg_clustcryst >= 0
    nn_c = nn_c[igood]
    avg_clustcryst = avg_clustcryst[igood]

    sites_cryst = np.zeros_like(avg_clustcryst)

    for k, n in enumerate(nn_c):
        sites_cryst[k] = np.mean(np.load(sitcryst_dir + f'sites_crystallinities_crystalline_all-{n}.npy'))
    

    # frac_cryst_atoms = np.load(strucdir + f'frac_cryst_atoms_{st}.npy')
    # frac_cryst_atoms = frac_cryst_atoms[nn_c]
    
    # igood = (avg_clustcryst >= 0) * (frac_cryst_atoms > 0) # I probably should not have to be filtering -1s... oh well
    # nn_c = nn_c[igood]
    # avg_clustcryst = avg_clustcryst[igood]
    # frac_cryst_atoms = frac_cryst_atoms[igood]

 
    # ax.scatter(frac_cryst_atoms, avg_clustcryst, s=20.0,c=c,alpha=0.7,zorder=1,label=lbl)
    ax.scatter(sites_cryst, avg_clustcryst, s=20.0,c=c,alpha=0.7,zorder=1,label=lbl)

ax.plot(np.linspace(0,1,100),np.linspace(0,1,100),'k--',lw=0.9)
# ax.set_xlabel('Fraction of crystalline atoms')
ax.set_xlabel('Average site crystallinity')
ax.set_ylabel('Conducting cluster crystallinity')
ax.legend()
plt.show()