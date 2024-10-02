#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from qcnico.plt_utils import MAC_ensemble_colours, setup_tex



def get_cluster_crystallinities(datadir, T):
    "For a given structure, obtain load the crystallinities of all the sites in its percolating cluster, at temperature T."
    nn = np.sort([int(d.split('-')[1]) for d in os.listdir(datadir)])
    lens = np.zeros(nn.shape[0],dtype='int')
    npys = [datadir + f'sample-{n}/clust_cryst-{T}K.npy' for n in nn]
    clust_crysts = np.load(npys[0])
    lens[0] = clust_crysts.shape[0]
    for k, f in enumerate(npys[1:]):
        new_cc = np.load(f)
        clust_crysts = np.hstack([clust_crysts, new_cc])
        lens[k+1] = new_cc.shape[0] 

    return nn, clust_crysts, lens

def create_struc_crystallinities_array(structype, nn, n_ccs):
    """Given an ensemble name,  a list of structure indices `nn`, and a list of number of conducting sites per structure `n_ccs`; 
    generate a list associating each conducting site in the ensemble to the fraction of crystalline atoms of its underlying structure.
    
    This is mean to prepare the crystallinity data for the scatter plot produced in this script."""

    strucdir = os.path.expanduser('~/Desktop/simulation_outputs/structural_characteristics_MAC/nb_cryst_atoms/')  
    frac_cryst_atoms = np.load(strucdir + f'nb_cryst_atoms_{structype}.npy')

    if structype == '40x40':
        nn -= 1 # 40x40 are indexed 1 --> 300

    cryst_fracs = np.zeros(n_ccs.sum())
    k = 0
    for n, n_cc in zip(nn, n_ccs):
        if frac_cryst_atoms[n] == 0: print(f'Struc # {n} of ensemble {structype} has 0 crystalline atoms!')
        cryst_fracs[k:k+n_cc] = frac_cryst_atoms[n]
    
    return cryst_fracs


def average_cluster_crystallinity(datadir, temps):
    """For each structure of a given ensemble (specified by `datadir`), returns the average crystallinity of all sites contained in 
    a conducting cluster.
    The avergae is performed over all sites in clusters from all temperatures."""
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
T = 280

percdir = os.path.expanduser('~/Desktop/simulation_outputs/percolation/')
strucdir = os.path.expanduser('~/Desktop/simulation_outputs/structural_characteristics_MAC/nb_cryst_atoms/') 


setup_tex()
fig, ax = plt.subplots()


for st, rmax, c, lbl in zip(ensembles,max_radii,clrs,official_labels):
    print(f'***** {st} *****')
    clustcryst_dir = percdir + f'{st}/electronic_crystallinity/cluster_crystallinities_rmax_{rmax}/'
    sitcryst_dir = percdir + f'{st}/electronic_crystallinity/sites_crystallinities_crystalline_{st}/all_sites/'
    # nn_c, avg_clustcryst = average_cluster_crystallinity(clustcryst_dir, temps)
    nn_c, clustcryst, nb_cond_sites = get_cluster_crystallinities(clustcryst_dir, T)

    frac_cryst_atoms = create_struc_crystallinities_array(st, nn_c, nb_cond_sites)

 

    

    
    igood = (frac_cryst_atoms >= 0) * (clustcryst >= 0) # I probably should not have to be filtering -1s... oh well
    # nn_c = nn_c[igood]
    clustcryst = clustcryst[igood]
    frac_cryst_atoms = frac_cryst_atoms[igood]

 
    ax.scatter(frac_cryst_atoms, clustcryst, s=2.0,c=c,alpha=0.7,zorder=1,label=lbl)

ax.plot(np.linspace(0,1,100),np.linspace(0,1,100),'k--',lw=0.9)
ax.set_xlabel('Fraction of crystalline atoms')
ax.set_ylabel('Conducting site crystallinity')
ax.legend()
plt.show()