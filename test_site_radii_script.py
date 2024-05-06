#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.qcplots import plot_atoms, plot_MO
from qcnico.coords_io import read_xsf
from MOs2sites import assign_AOs, get_MO_loc_centers_opt
from matplotlib import rcParams

nsample = 127

percdir = f'/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tempdot6/percolate_output/zero_field/virt_100x100_gridMOs/sample-{nsample}/'
Mdir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tempdot6/MOs_ARPACK/virtual/'
posdir = '/Users/nico/Desktop/simulation_outputs/MAC_structures/Ata_structures/tempdot6/relaxed/'

MO_inds = [0,3,-2,-1]

cc = np.load(percdir + 'cc.npy')
ii = np.load(percdir + 'ii.npy')
M = np.load(Mdir + f'MOs_ARPACK_bigMAC-{nsample}.npy') 
pos,_ = read_xsf(posdir + f'tempdot6n{nsample}_relaxed.xsf')

MO_inds = np.unique(ii)[MO_inds]
MO_inds = [3]
print(MO_inds)

pos = pos[:,:2]
sites, rho, xedges, yedges = get_MO_loc_centers_opt(pos, M, 3, nbins=100,threshold_ratio=0.4, shift_centers=False,min_distance=20)
nsites = sites.shape[0]
print('nsites = ',nsites)

# for n in MO_inds:
#     # site_inds = (ii == n).nonzero()[0]
#     # print(site_inds)
#     # sites = cc[site_inds]
    
#     fig,ax = plt.subplots()

#     fig, ax = plot_MO(pos,M,n,dotsize=0.5,show=False,plt_objs=(fig,ax),usetex=True,scale_up=50)
#     ax.scatter(sites[:,0],sites[:,1],marker='*',color='r',s=80,edgecolors='k',label='hopping sites')
#     plt.legend()
#     plt.show()

cyc = rcParams['axes.prop_cycle'] #default plot colours are stored in this `cycler` type object
print(len(cyc))


for n in MO_inds:
    # site_inds = (ii == n).nonzero()[0]
    # nsites  = site_inds.shape[0]
    # sites = cc[site_inds]
    print('nsites = ',nsites)
    colors = [d['color'] for d in list(cyc[0:nsites])]
    print('len(colors) = ', len(colors))
    psi = M[:,n]

    cluster_centers,labels = assign_AOs(pos,sites,psi=psi)
    print(labels.shape)
    print(np.max(labels))

    # atom_clrs  = [colors[k] for k in labels]
    atom_clrs = [0] * labels.shape[0]
    print(len(atom_clrs))
    for k,l in enumerate(labels):
        # print(k,l)
        atom_clrs[k] = colors[l]
    cluster_clrs = colors
    cc_clrs = colors
    
    print('cc_clrs = ',cc_clrs)
    print('cluster_clrs = ',cluster_clrs)
    
    fig,ax = plt.subplots()

    fig, ax = plot_atoms(pos,dotsize=0.1,show=False,plt_objs=(fig,ax),usetex=False,colour=atom_clrs,zorder=1)
    ax.scatter(*cluster_centers.T,marker='^',c=cluster_clrs,edgecolors='k',s=60.0,zorder=2,label='cluster centres')
    ax.scatter(*sites.T,marker='*',c=cc_clrs,edgecolors='k',s=60.0,zorder=3,label='hopping sites')
    ax.set_title(f'MO \# {n}')
    plt.legend()
    plt.show()

    fig,ax = plt.subplots()

    fig, ax = plot_MO(pos,M,n,dotsize=0.5,show=False,plt_objs=(fig,ax),usetex=True,scale_up=50)
    ax.scatter(*cluster_centers.T,marker='^',c='r',edgecolors='k',s=60.0,zorder=2,label='cluster centres')
    # ax.scatter(*sites.T,marker='*',c=cc_clrs,edgecolors='k',s=60.0,zorder=3)
    ax.set_title(f'MO \# {n}')
    plt.legend()
    plt.show()

