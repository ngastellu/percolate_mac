#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.qcplots import plot_atoms, plot_MO
from qcnico.coords_io import read_xsf
from qcnico.plt_utils import get_cm, histogram
from MOs2sites import assign_AOs, get_MO_loc_centers_opt, assign_AOs_naive, site_radii
from matplotlib import rcParams
from scipy.spatial import Voronoi, voronoi_plot_2d


nsample = 127

percdir = f'/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tempdot6/percolate_output/zero_field/virt_100x100_gridMOs/sample-{nsample}/'
Mdir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tempdot6/MOs_ARPACK/virtual/'
posdir = '/Users/nico/Desktop/simulation_outputs/MAC_structures/Ata_structures/tempdot6/relaxed/'

# MO_inds = [0,3,-2,-1]

# cc = np.load(percdir + 'cc.npy')
# ii = np.load(percdir + 'ii.npy')
M = np.load(Mdir + f'MOs_ARPACK_bigMAC-{nsample}.npy') 
pos,_ = read_xsf(posdir + f'tempdot6n{nsample}_relaxed.xsf')

# MO_inds = np.unique(ii)[MO_inds]
MO_ind = 173
# print(MO_inds)

pos = pos[:,:2]
sites, rho, xedges, yedges = get_MO_loc_centers_opt(pos, M, MO_ind, nbins=100,threshold_ratio=0.3, shift_centers=False,min_distance=30)
nsites = sites.shape[0]
print('nsites = ',nsites)

    
fig,ax = plt.subplots()

fig, ax = plot_MO(pos,M,MO_ind,dotsize=0.3,show=False,plt_objs=(fig,ax),usetex=True,scale_up=50)
ax.scatter(sites[:,0],sites[:,1],marker='*',color='r',s=80,edgecolors='k',label='hopping sites')
plt.legend()
plt.show()

cyc = rcParams['axes.prop_cycle'] #default plot colours are stored in this `cycler` type object
print(len(cyc))


if nsites <= 10:
    colors = [d['color'] for d in list(cyc[0:nsites])]
else:
    colors = get_cm(np.arange(nsites),'hsv',max_val=0.8)
        
print('len(colors) = ', len(colors))
psi = M[:,MO_ind]

cluster_centers,labels = assign_AOs(pos,sites,psi=psi)
labels2 = assign_AOs_naive(pos, sites)

centers_kmeans, radii_kmeans = site_radii(pos, M, MO_ind, labels)

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
ax.set_title(f'MO \# {MO_ind}')
plt.legend()
plt.show()


if nsites > 1:

    fig,ax = plt.subplots()

    vor = Voronoi(cluster_centers)

    fig, ax = plot_MO(pos,M,MO_ind,dotsize=0.5,show=False,plt_objs=(fig,ax),usetex=True,scale_up=50,scale_up_threshold=0.0015)
    ax.scatter(*cluster_centers.T,marker='^',c='r',edgecolors='k',s=60.0,zorder=2,label='cluster centres')
    fig = voronoi_plot_2d(vor, ax, line_color='white', show_vertices=False, show_points=False)
    ax.scatter(*sites.T,marker='*',c='r',edgecolors='k',s=60.0,zorder=3,label='hopping sites')
    ax.set_title(f'Partition of $|\psi_{{{MO_ind}}}\\rangle$ (k-means)')
    fig = voronoi_plot_2d(vor, ax, line_color='snow', show_vertices=False, show_points=False)
    ax.set_xlim([0,400])
    ax.set_ylim([0,400])
    plt.legend()
    plt.show()

    fig,ax = plt.subplots()

    vor = Voronoi(sites)

    fig, ax = plot_MO(pos,M,MO_ind,dotsize=0.5,show=False,plt_objs=(fig,ax),usetex=True,scale_up=50)
    ax.scatter(*cluster_centers.T,marker='^',c='r',edgecolors='k',s=60.0,zorder=2,label='cluster centres')
    ax.scatter(*sites.T,marker='*',c='r',edgecolors='k',s=60.0,zorder=3,label='hopping sites')
    fig = voronoi_plot_2d(vor, ax, line_color='snow', show_vertices=False, show_points=False)
    ax.set_title(f'Partition MO $|\psi_{{{MO_ind}}}\\rangle$ (Voronoi)')
    ax.set_xlim([0,400])
    ax.set_ylim([0,400])
    plt.legend()
    plt.show()

# sub_densities = np.zeros(nsites)


threshold_ratio_psi = 10.0
psi = M[:,MO_ind] ** 2
nbins = 200
_, bins = np.histogram(psi, bins=nbins)
centers = 0.5 * (bins[:-1] + bins[1:])
clrs = get_cm(centers,'plasma')
fig, ax = plt.subplots() 
fig, ax = histogram(psi,nbins=nbins,plt_objs=(fig,ax),log_counts=True,show=False,plt_kwargs={'color':clrs})
# ax.axvline(x=np.mean(psi),ymin=0,ymax=1,c='k',lw=0.8,ls='--')
# ax.axvline(x=np.mean(psi)+threshold_ratio_psi*np.std(psi),ymin=0,ymax=1,c='k',lw=0.8,ls='--',label='cutoff')
ax.axvline(x=0.001,ymin=0,ymax=1,c='k',lw=0.8,ls='--',label='cutoff')
ax.set_xlabel(f'$|\langle\\varphi_n|\psi_{{{MO_ind}}}\\rangle|^2$')
plt.legend()
plt.show()

# threshold = np.mean(psi) + threshold_ratio_psi * np.std(psi)
# psi_cut = np.copy(psi)
# psi_cut[psi < threshold] = 0

# sub_densities = np.array([np.sum(psi_cut[labels==k])/(labels == k).sum() for k in range(nsites)])
# histogram(sub_densities,nbins=5)


# cluster_densities_clrs = get_cm(sub_densities,'plasma',max_val=0.9)
# atom_clrs  = [cluster_densities_clrs[k] for k in labels]


# print(sub_densities)

# fig, ax = plot_atoms(pos,dotsize=0.1,show=False,usetex=False,colour=atom_clrs,zorder=1)
# ax.scatter(*cluster_centers.T,marker='^',c=cluster_clrs,edgecolors='k',s=60.0,zorder=2,label='cluster centres')
# ax.scatter(*sites.T,marker='*',c=cc_clrs,edgecolors='k',s=60.0,zorder=3,label='hopping sites')
# ax.set_title(f'MO \# {MO_ind}')
# plt.legend()
# plt.show()


# good_bools = (sub_densities >= 0.00002)

# if good_bools.shape[0] > 0 and good_bools.shape[0] < sub_densities.shape[0]:

#     cluster_center_dists = np.linalg.norm(cluster_centers[:,None,:] - cluster_centers,axis=2)
#     np.fill_diagonal(cluster_center_dists, 10000)

#     igood_clusters = good_bools.nonzero()[0]
#     ibad_clusters = (~good_bools).nonzero()[0]

#     print(igood_clusters)
#     print(ibad_clusters)

#     labels_map = np.arange(nsites)

#     good_cluster_centers = cluster_centers[igood_clusters]

#     for n in ibad_clusters:
#         r = cluster_centers[n,:]
#         dists = np.linalg.norm(good_cluster_centers - r[None,:],axis=1)
#         iclosest_cc = np.argmin(dists)
#         labels_map[n] = iclosest_cc
#         print(f'{n} ---> {iclosest_cc}')

#     for k in range(labels.shape[0]):
#         old_l = labels[k]
#         labels[k] = labels_map[old_l]

#     n = MO_ind

#     cluster_centers = cluster_centers[~(np.all(cluster_centers == 1000,axis=1))]
#     nsites = cluster_centers.shape[0]
#     print(nsites)
#     sub_densities = np.array([np.sum(psi[labels==k]) for k in range(nsites)])
#     cluster_densities_clrs = get_cm(sub_densities,'plasma',max_val=0.9)
#     atom_clrs  = [cluster_densities_clrs[k] for k in labels]

#     fig, ax = plot_atoms(pos,dotsize=0.1,show=False,usetex=False,colour=atom_clrs,zorder=1)
#     ax.scatter(*cluster_centers.T,marker='^',c='r',edgecolors='k',s=60.0,zorder=2,label='cluster centres')
#     ax.scatter(*sites.T,marker='*',c='r',edgecolors='k',s=60.0,zorder=3,label='hopping sites')
#     ax.set_title(f'MO \# {n}')
#     plt.legend()
#     plt.show()



# fig, ax = plot_MO(pos,M,MO_ind,dotsize=0.5,show=False,usetex=False,scale_up=50,zorder=1,show_rgyr=False,show_COM=False)
# ax.scatter(*cluster_centers.T,marker='^',c='r',edgecolors='k',s=60.0,zorder=2,label='cluster centres')
# ax.scatter(*sites.T,marker='*',c='r',edgecolors='k',s=60.0,zorder=3,label='hopping sites')
# ax.set_title(f'MO \# {MO_ind}')
# plt.legend()
# plt.show()

 