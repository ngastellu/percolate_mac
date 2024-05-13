#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.qcplots import plot_atoms, plot_MO
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from qcnico.plt_utils import get_cm, histogram
from MOs2sites import assign_AOs, get_MO_loc_centers_opt, assign_AOs_naive, site_radii
from matplotlib import rcParams
from scipy.spatial import Voronoi, voronoi_plot_2d


nsample = 42

# percdir = f'/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tempdot5/percolate_output/zero_field/virt_100x100_gridMOs/sample-{nsample}/'
Mdir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/MOs_ARPACK/virtual/'
posdir = '/Users/nico/Desktop/simulation_outputs/MAC_structures/pCNN/bigMAC_40x40_ensemble/'

# MO_inds = [0,3,-2,-1]

# cc = np.load(percdir + 'cc.npy')
# ii = np.load(percdir + 'ii.npy')
M = np.load(Mdir + f'MOs_ARPACK_bigMAC-{nsample}.npy') 
# pos = np.load(posdir + f'coords-{nsample}.npy')
pos,_ = read_xsf(posdir + f'bigMAC-{nsample}_relaxed.xsf')
pos = remove_dangling_carbons(pos,1.8)

# MO_inds = np.unique(ii)[MO_inds]
# MO_inds = [58,122,173]
# print(MO_inds)

pos = pos[:,:2]

# MO_inds = np.random.randint(M.shape[1], size=10)
MO_inds = [71,79,92]

for MO_ind in MO_inds:

    print(f'********** {MO_ind} **********')

    sites, *_ = get_MO_loc_centers_opt(pos, M, MO_ind, nbins=100, threshold_ratio=0.3   ,shift_centers=True,min_distance=20.0)

    psi = M[:,MO_ind]
    centers_kmeans, labels_kmeans = assign_AOs(pos, sites, psi,psi_pow=4)
    centers_kmeans_threshold, labels_kmeans_threshold = assign_AOs(pos, sites, psi,psi_pow=4,density_threshold=0.001)

    labels_naive = assign_AOs_naive(pos,sites)

    sites_kmeans, radii_kmeans = site_radii(pos, M, MO_ind, labels_kmeans)
    # sites_kmeans_hl, radii_kmeans_hl = site_radii(pos, M, MO_ind, labels_kmeans, hyperlocal=True)

    # sites_kmeans_threshold, radii_kmeans_threshold = site_radii(pos, M, MO_ind, labels_kmeans_threshold)
    # sites_kmeans_threshold_hl, radii_kmeans_threshold_hl = site_radii(pos, M, MO_ind, labels_kmeans_threshold, hyperlocal=True)

    # ------ k-means approach: no threshold -------


    # print('k-means cluster centers = ', centers_kmeans[np.argsort(np.linalg.norm(centers_kmeans,axis=1)),:])
    print('Final localisation sites and radii (k-means, no threshold):')
    for r, a in zip(sites_kmeans,radii_kmeans):
        print(f'{r} --> {a}')

    print('----------')
    # fig,axs = plt.subplots(1,2,sharex=True,sharey=True) 
    fig, ax = plt.subplots()
    fig, ax = plot_MO(pos,M,MO_ind,dotsize=0.5,show=False,plt_objs=(fig,ax),usetex=True,scale_up=50,title=f'Partition of $|\psi_{{{MO_ind}}}\\rangle$ (k-means)',show_cbar=False)
    ax.scatter(*centers_kmeans.T,marker='^',c='r',edgecolors='k',s=60.0,zorder=2,label='cluster centres')
    if sites.shape[0] > 2:
        vor = Voronoi(centers_kmeans)
        fig = voronoi_plot_2d(vor, ax, line_color='white', show_vertices=False, show_points=False)
    else:
        print('!!!! Skipping Voronoi; only 1 site !!!!')
    ax.scatter(*sites_kmeans.T,marker='h',c='r',edgecolors='k',s=60.0,zorder=4,label='sites')
    # ax.scatter(*sites_kmeans_hl.T,marker='h',c='limegreen',edgecolors='k',s=60.0,zorder=4,label='final sites (hyperlocal)')
    for k in range(sites_kmeans.shape[0]): 
        print(sites_kmeans[k,:])
        # print(sites_kmeans_hl[k,:])
        loc_circle = plt.Circle(sites_kmeans[k,:], radii_kmeans[k], fc='none', ec='r', ls='--', lw=1.0,zorder=4)
        ax.add_patch(loc_circle)
        # loc_circle = plt.Circle(sites_kmeans_hl[k], radii_kmeans_hl[k], fc='none', ec='limegreen', ls='--', lw=1.0,zorder=4)
        # ax.add_patch(loc_circle)
    ax.set_xlim([0,400])
    ax.set_ylim([0,400])
    plt.legend()
    plt.show()

    # # ------ k-means approach: with threshold -------


    # # print('k-means cluster centers = ', centers_kmeans_threshold[np.argsort(np.linalg.norm(centers_kmeans_threshold,axis=1)),:])
    # # print('sites centers = ', sites_kmeans_threshold[np.argsort(np.linalg.norm(sites_kmeans_threshold,axis=1)),:])
    # # print('----------')
    # fig,ax = plt.subplots() 
    # fig, ax = plot_MO(pos,M,MO_ind,dotsize=0.5,show=False,plt_objs=(fig,ax),usetex=True,scale_up=50,title=f'Partition of $|\psi_{{{MO_ind}}}\\rangle$ (k-means, thresh)')
    # ax.scatter(*centers_kmeans_threshold.T,marker='^',c='r',edgecolors='k',s=60.0,zorder=2,label='cluster centres')
    # if sites.shape[0] > 1:
    #     vor = Voronoi(centers_kmeans_threshold)
    #     fig = voronoi_plot_2d(vor, ax, line_color='white', show_vertices=False, show_points=False)
    # else:
    #     print('!!!! Skipping Voronoi; only 1 site !!!!')
    # ax.scatter(*sites_kmeans_threshold.T,marker='h',c='r',edgecolors='k',s=60.0,zorder=4,label='final sites')
    # ax.scatter(*sites_kmeans_threshold_hl.T,marker='h',c='limegreen',edgecolors='k',s=60.0,zorder=4,label='final sites (hyperlocal)')
    # for k in range(sites_kmeans_threshold.shape[0]): 
    #     print(sites_kmeans_threshold[k,:])
    #     print(sites_kmeans_threshold_hl[k,:])
    #     loc_circle = plt.Circle(sites_kmeans_threshold[k,:], radii_kmeans_threshold[k], fc='none', ec='r', ls='--', lw=1.0,zorder=4)
    #     ax.add_patch(loc_circle)
    #     loc_circle = plt.Circle(sites_kmeans_threshold_hl[k], radii_kmeans_threshold_hl[k], fc='none', ec='limegreen', ls='--', lw=1.0,zorder=4)
    #     ax.add_patch(loc_circle)
    # ax.set_xlim([0,400])
    # ax.set_ylim([0,400])
    # # plt.legend()
    # plt.show()

    # ------ Voronoi only: 'naive' approach ------


    # sites_naive, radii_naive = site_radii(pos, M, MO_ind, labels_naive)
    # # sites_naive_hl, radii_naive_hl = site_radii(pos, M, MO_ind, labels_naive, hyperlocal=True)

    # # fig,ax = plt.subplots()
    # fig, axs[1] = plot_MO(pos,M,MO_ind,dotsize=0.5,show=False,plt_objs=(fig,axs[1]),usetex=True,scale_up=50,title=f'Partition MO $|\psi_{{{MO_ind}}}\\rangle$ (Voronoi)',show_cbar=False)
    # axs[1].scatter(*sites.T,marker='*',c='r',edgecolors='k',s=60.0,zorder=3,label='hopping sites')
    # if sites.shape[0] > 2:
    #     vor = Voronoi(sites_naive)
    #     fig = voronoi_plot_2d(vor, axs[1], line_color='white', show_vertices=False, show_points=False)
    # else:
    #     print('!!!! Skipping Voronoi; less than 3 sites !!!!')
    # axs[1].scatter(*sites_naive.T,marker='h',c='r',edgecolors='k',s=60.0,zorder=4,label='final sites')
    # # axs[1].scatter(*sites_naive_hl.T,marker='h',c='limegreen',edgecolors='k',s=60.0,zorder=4,label='final sites (hyperlocal)')
    # for k in range(sites_naive.shape[0]): 
    #     loc_circle = plt.Circle(sites_naive[k,:], radii_naive[k], fc='none', ec='r', ls='--', lw=1.0,zorder=4)
    #     axs[1].add_patch(loc_circle)
    #     # # loc_circle = plt.Circle(sites_naive_hl[k], radii_naive_hl[k], fc='none', ec='limegreen', ls='--', lw=1.0,zorder=4)
    #     # axs[1].add_patch(loc_circle)
    # axs[1].set_xlim([0,400])
    # axs[1].set_ylim([0,400])
    # # plt.legend()
    # plt.show()

    # ------ Voronoi only: 'naive' approach; with threshold ------

    # sites_naive, radii_naive = site_radii(pos, M, MO_ind, labels_naive,density_threshold=0.001)
    # sites_naive_hl, radii_naive_hl = site_radii(pos, M, MO_ind, labels_naive, hyperlocal=True,density_threshold=0.001)

    # fig,ax = plt.subplots()
    # fig, ax = plot_MO(pos,M,MO_ind,dotsize=0.5,show=False,plt_objs=(fig,ax),usetex=True,scale_up=50,title=f'Partition MO $|\psi_{{{MO_ind}}}\\rangle$ (Voronoi, t)')
    # ax.scatter(*sites.T,marker='*',c='r',edgecolors='k',s=60.0,zorder=3,label='hopping sites')
    # if sites.shape[0] > 1:
    #     vor = Voronoi(sites)
    #     fig = voronoi_plot_2d(vor, ax, line_color='white', show_vertices=False, show_points=False)
    # else:
    #     print('!!!! Skipping Voronoi; only 1 site !!!!')
    # ax.scatter(*sites_naive.T,marker='h',c='r',edgecolors='k',s=60.0,zorder=4,label='final sites')
    # ax.scatter(*sites_naive_hl.T,marker='h',c='limegreen',edgecolors='k',s=60.0,zorder=4,label='final sites (hyperlocal)')
    # for k in range(sites_naive.shape[0]): 
    #     loc_circle = plt.Circle(sites_naive[k,:], radii_naive[k], fc='none', ec='r', ls='--', lw=1.0,zorder=4)
    #     ax.add_patch(loc_circle)
    #     loc_circle = plt.Circle(sites_naive_hl[k], radii_naive_hl[k], fc='none', ec='limegreen', ls='--', lw=1.0,zorder=4)
    #     ax.add_patch(loc_circle)
    # ax.set_xlim([0,400])
    # ax.set_ylim([0,400])
    # # plt.legend()
    # plt.show()


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

 