#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from qcnico.plt_utils import MAC_ensemble_colours, setup_tex




def get_successful_istruc(cc_datadir):
    """Obtain list of structure indices for which a specific percolation calculation ran successfully (and for which we therefore have cluster crystallinity data)."""
    
    return np.sort([int(d.split('-')[1]) for d in os.listdir(cc_datadir)])


def get_cluster_crystallinities(cc_datadir, n, T):
    return np.load(os.path.join(cc_datadir, f'sample-{n}/clust_cryst-{T}K.npy'))


def get_cluster_crystallinities_ensemble(datadir, T):
    "For a given ensemble, load the crystallinities of all the sites in the percolating cluster of all structures, at temperature T."
    nn = get_successful_istruc(datadir)
    lens = np.zeros(nn.shape[0],dtype='int')
    npys = [datadir + f'sample-{n}/clust_cryst-{T}K.npy' for n in nn]
    clust_crysts = np.load(npys[0])
    lens[0] = clust_crysts.shape[0]
    for k, f in enumerate(npys[1:]):
        new_cc = np.load(f)
        clust_crysts = np.hstack([clust_crysts, new_cc])
        lens[k+1] = new_cc.shape[0] 

    return nn, clust_crysts, lens

def get_struc_crystallinities(structype, nn, n_ccs):
    strucdir = os.path.expanduser('~/Desktop/simulation_outputs/structural_characteristics_MAC/fraction_cryst_atoms/')  
    frac_cryst_atoms = np.load(strucdir + f'frac_cryst_atoms_{structype}.npy')

    cryst_fracs = np.zeros(n_ccs.sum())
    k = 0
    for n, n_cc in zip(nn, n_ccs):
        if frac_cryst_atoms[n] == 0: print(f'Struc # {n} of ensemble {structype} has 0 crystalline atoms!')
        cryst_fracs[k:k+n_cc] = frac_cryst_atoms[n]
        k += n_cc
    
    return cryst_fracs

def make_plot_array(structype, motype, T):
    """Given an ensemble name,  a list of structure indices `nn`, and a list of number of conducting sites per structure `n_ccs`; 
    generate a list associating each conducting site in the ensemble to the fraction of crystalline atoms of its underlying structure.
    
    This is mean to prepare the crystallinity data for the scatter plot produced in this script."""

    if structype == '40x40':
        rmax = 18.03
    elif structype == 'tempdot6':
        rmax = 121.2
    elif structype == 'tempdot5':
        rmax = 198.69
    else:
        print(f'Invalid structure type {structype}. Returning 0 awkwardly.')
    
    percdir = os.path.expanduser('~/Desktop/simulation_outputs/percolation/')
    strucdir = os.path.expanduser('~/Desktop/simulation_outputs/structural_characteristics_MAC/nb_cryst_atoms/') 

    cc_datadir = percdir + f'{st}/electronic_crystallinity/cluster_crystallinities_sites_gammas/cluster_crystallinities_rmax_{rmax}_sites_gammas_{motype}/'
    nn, clust_crysts, n_cond_sites = get_cluster_crystallinities_ensemble(cc_datadir,T)

    if structype == '40x40':
        nn -= 1 # indices from PixelCNN are 1-indexed

    frac_cryst_atoms = get_struc_crystallinities(structype, nn, n_cond_sites)

    ccfilter = clust_crysts >= 0
    
    return frac_cryst_atoms[ccfilter], clust_crysts[ccfilter]


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
clrs = MAC_ensemble_colours()
temps = np.arange(180,440,10)
T = 300

motype = 'virtual'


setup_tex(fontsize=70)
rcParams['figure.figsize'] = [8,7]
fig, ax = plt.subplots()


for st, c, lbl in zip(ensembles,clrs,official_labels):
    print(f'***** {st} *****')

    frac_cryst_atoms, clustcryst = make_plot_array(st, motype, T)

    print('% of 0 cryst condistes = ', (clustcryst < 1e-10).sum()*100/clustcryst.shape[0])
 
    ax.scatter(frac_cryst_atoms, clustcryst, s=30.0,c=c,alpha=0.7,zorder=1,label=lbl)

# ax.plot(np.linspace(1/6,1,1000),1.0/np.linspace(1/6,1,1000),'k--',lw=0.9)
ax.plot(np.linspace(0,1,100),np.linspace(0,1,100),'k--',lw=3.0)
ax.set_xlabel('Fraction of crystalline atoms $\phi_c$')
ax.set_ylabel('Conducting site crystallinity $\chi$\n')
ax.tick_params('x',which='major',length=10,width=1.6)
ax.tick_params('y',which='major',length=10,width=1.6)
ax.set_aspect('equal')
# ax.legend()

# ax.set_title(f'Conducting clusters of sites from {motype} MOs at $T = {T}$K')

plt.show()