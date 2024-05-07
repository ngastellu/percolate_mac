#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.qcplots import plot_atoms, plot_MO
from qcnico.coords_io import read_xsf
from qcnico.plt_utils import get_cm, histogram, multiple_histograms
from MOs2sites import assign_AOs, get_MO_loc_centers_opt
from matplotlib import rcParams
from scipy.spatial import Voronoi, voronoi_plot_2d


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
sites, rho, xedges, yedges = get_MO_loc_centers_opt(pos, M, 3, nbins=100,threshold_ratio=0.3, shift_centers=False,min_distance=20)
nsites = sites.shape[0]
print('nsites = ',nsites)

for n in MO_inds:
    # site_inds = (ii == n).nonzero()[0]
    # print(site_inds)
    # sites = cc[site_inds]
    
    fig,ax = plt.subplots()

    fig, ax = plot_MO(pos,M,n,dotsize=0.5,show=False,plt_objs=(fig,ax),usetex=True,scale_up=50)
    ax.scatter(sites[:,0],sites[:,1],marker='*',color='r',s=80,edgecolors='k',label='hopping sites')
    plt.legend()
    plt.show()

cyc = rcParams['axes.prop_cycle'] #default plot colours are stored in this `cycler` type object
print(len(cyc))


for n in MO_inds:
    # site_inds = (ii == n).nonzero()[0]
    # nsites  = site_inds.shape[0]
    # sites = cc[site_inds]
    print('nsites = ',nsites)
    if nsites <= 10:
        colors = [d['color'] for d in list(cyc[0:nsites])]
    # elif nsites <= 20:
    #     colors = get_cm(np.arange(nsites),'tab20')
    else:
        colors = get_cm(np.arange(nsites),'hsv',max_val=0.8)
        
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

    vor = Voronoi(cluster_centers)

    fig, ax = plot_MO(pos,M,n,dotsize=0.5,show=False,plt_objs=(fig,ax),usetex=True,scale_up=50)
    ax.scatter(*cluster_centers.T,marker='^',c='r',edgecolors='k',s=60.0,zorder=2,label='cluster centres')
    fig = voronoi_plot_2d(vor, ax, line_color='white', show_vertices=False, show_points=False)
    # ax.scatter(*sites.T,marker='*',c=cc_clrs,edgecolors='k',s=60.0,zorder=3)
    ax.set_title(f'MO \# {n}')
    plt.legend()
    plt.show()


# sub_densities = np.zeros(nsites)

psi = M[:,n] ** 2

threshold = np.mean(psi) + 5.0 * np.std(psi)
psi[psi < threshold] = 0

sub_densities = np.array([np.sum(psi[labels==k])/(labels == k).sum() for k in range(nsites)])
histogram(sub_densities,nbins=5)


cluster_densities_clrs = get_cm(sub_densities,'plasma',max_val=0.9)
atom_clrs  = [cluster_densities_clrs[k] for k in labels]


print(sub_densities)

fig, ax = plot_atoms(pos,dotsize=0.1,show=False,usetex=False,colour=atom_clrs,zorder=1)
ax.scatter(*cluster_centers.T,marker='^',c=cluster_clrs,edgecolors='k',s=60.0,zorder=2,label='cluster centres')
ax.scatter(*sites.T,marker='*',c=cc_clrs,edgecolors='k',s=60.0,zorder=3,label='hopping sites')
ax.set_title(f'MO \# {n}')
plt.legend()
plt.show()


cluster_center_dists = np.linalg.norm(cluster_centers[:,None,:] - cluster_centers,axis=2)
np.fill_diagonal(cluster_center_dists, 10000)

good_bools = (sub_densities >= 0.00002)
igood_clusters = good_bools.nonzero()[0]
ibad_clusters = (~good_bools).nonzero()[0]

labels_map = np.arange(nsites)

good_cluster_centers = cluster_centers[igood_clusters]

for n in ibad_clusters:
    r = cluster_centers[n,:]
    dists = np.linalg.norm(good_cluster_centers - r[None,:],axis=1)
    iclosest_cc = np.argmin(dists)
    labels_map[n] = iclosest_cc
    print(f'{n} ---> {iclosest_cc}')

for k in range(labels.shape[0]):
    old_l = labels[k]
    labels[k] = labels_map[old_l]

n = 3

cluster_centers = cluster_centers[~(np.all(cluster_centers == 1000,axis=1))]
nsites = cluster_centers.shape[0]
print(nsites)
sub_densities = np.array([np.sum(psi[labels==k]) for k in range(nsites)])
cluster_densities_clrs = get_cm(sub_densities,'plasma',max_val=0.9)
atom_clrs  = [cluster_densities_clrs[k] for k in labels]

fig, ax = plot_atoms(pos,dotsize=0.1,show=False,usetex=False,colour=atom_clrs,zorder=1)
ax.scatter(*cluster_centers.T,marker='^',c='r',edgecolors='k',s=60.0,zorder=2,label='cluster centres')
ax.scatter(*sites.T,marker='*',c='r',edgecolors='k',s=60.0,zorder=3,label='hopping sites')
ax.set_title(f'MO \# {n}')
plt.legend()
plt.show()



fig, ax = plot_MO(pos,M,n,dotsize=0.5,show=False,usetex=False,scale_up=50,zorder=1,show_rgyr=True,show_COM=True)
ax.scatter(*cluster_centers.T,marker='^',c='r',edgecolors='k',s=60.0,zorder=2,label='cluster centres')
ax.scatter(*sites.T,marker='*',c='r',edgecolors='k',s=60.0,zorder=3,label='hopping sites')
ax.set_title(f'MO \# {n}')
plt.legend()
plt.show()


MO_inds = [0,1,2,3,-2,-1]

for n in MO_inds:
    psi = np.abs(M[:,n])**2
    fig, ax = plt.subplots() 
    psi = np.abs(M[:,n])**2
    fig, ax = histogram(psi,nbins=200,plt_objs=(fig,ax),usetex=False,log_counts=True,show=False)
    ax.axvline(x=np.mean(psi),ymin=0,ymax=1,c='k',lw=0.8,ls='--')
    ax.axvline(x=np.mean(psi)+np.std(psi),ymin=0,ymax=1,c='k',lw=0.8,ls='--')
    ax.axvline(x=np.mean(psi)+2*np.std(psi),ymin=0,ymax=1,c='k',lw=0.8,ls='--')
    ax.axvline(x=np.mean(psi)+3*np.std(psi),ymin=0,ymax=1,c='k',lw=0.8,ls='--')
    plt.legend()
    plt.show()