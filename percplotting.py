#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl 
from qcnico import plt_utils
import qcnico.qchemMAC as qcm
from qcnico.graph_tools import adjacency_matrix_sparse



def plot_cluster(c,pos, M, adjmat,show_densities=False,dotsize=20, usetex=True, show=True, centers=None, rel_center_size=2.0, inds=None, plt_objs=None):
    pos = pos[:,:2]

    c = np.sort(list(c))
    if centers is None:
        centers = qcm.MO_com(pos,M,c)
        inds = c
    else:
        assert inds is not None, "[percolate.plot_cluster] If `centers` is passed, so must `inds`!"
        centers = centers[c,:]
        print(centers)
        
        
    
    if plt_objs is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt_objs

    if usetex:
        plt_utils.setup_tex()

    if show_densities:
        rho = np.sum(M[:,np.unique(inds)]**2,axis=1)
        ye = ax.scatter(pos.T[0], pos.T[1], c=rho, s=dotsize, cmap='plasma',zorder=1)
        cbar = fig.colorbar(ye,ax=ax,orientation='vertical')

    else:
        ax.scatter(pos.T[0], pos.T[1], c='k', s=dotsize)

    ax.scatter(*centers.T, marker='*', c='r', s = rel_center_size*dotsize,zorder=2)
    seen = set()
    for i in c:
        if i not in seen:
            n = np.sum(i > c) #gets relative index of i (i=global MO index; n=index of MO i in centers array)
            r1 = centers[n]
            neighbours = adjmat[i,:].nonzero()[0]
            print(neighbours)
            seen.update(neighbours)
            for j in neighbours:
                m = np.sum(j > c)
                r2 = centers[m]
                pts = np.vstack((r1,r2)).T
                ax.plot(*pts, 'r-', lw=0.7)
                nn = adjmat[j,:].nonzero()[0]
                seen.update(nn)
                for n in nn:
                    m = np.sum(n > c)
                    r3 = centers[m]
                    pts = np.vstack((r3,r2)).T
                    ax.plot(*pts, 'r-', lw=0.7)

    
    if show:
        plt.show()


def plot_cluster_brute_force(c,pos, M, adjmat,show_edges=True, show_densities=False,dotsize=20, usetex=True, show=True, centers=None, rel_center_size=2.0, inds=None, plt_objs=None):
    pos = pos[:,:2]
    N = pos.shape[0]

    c = np.sort(list(c))
    if centers is None:
        centers = qcm.MO_com(pos,M,c)
        inds = c
    else:
        # assert inds is not None, "[percolate.plot_cluster] If `centers` is passed, so must `inds`!"
        centers = centers[c,:]
        print(centers)
        
    if plt_objs is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt_objs

    if usetex:
        plt_utils.setup_tex()

    if show_densities:
        print(np.unique(inds))
        rho = np.log(np.sum(M[:,np.unique(inds)]**2,axis=1))
        sizes = np.ones(N) * dotsize
        sizes[rho > 0.002] *= 10
        ye = ax.scatter(pos.T[0], pos.T[1], c=rho, s=sizes, cmap='plasma',zorder=1)
        cbar = fig.colorbar(ye,ax=ax,orientation='vertical')

    else:
        ax.scatter(pos.T[0], pos.T[1], c='k', s=dotsize)

    # Draw sites
    # ax.scatter(*centers.T, marker='*', c='r', s = rel_center_size*dotsize,zorder=2)
    ax.set_aspect('equal')
    ax.set_xlabel("$x$ [\AA]")
    ax.set_ylabel("$y$ [\AA]")

    # Draw edges between each site and its neighbours
    if show_edges:
        for i in c:
            n = np.sum(i > c) #gets relative index of i (i=global MO index; n=index of MO i in centers array)
            r1 = centers[n]
            neighbours = adjmat[i,:].nonzero()[0] 
            for j in neighbours:
                m = np.sum(j>c)
                r2 = centers[m]
                pts = np.vstack((r1,r2)).T
                # ax.plot(*pts, 'r-', lw=0.7)
                ax.plot(*pts, 'r-', lw=1.0)
    
    if show:
        plt.show()

def plot_cluster_density(c,pos, M,dotsize=20, big_rho_threshold=0.002, rel_center_size=10.0, show_bonds=False, xbounds=None, ybounds=None, vmin=None, vmax=None, usetex=True, show=True, plt_objs=None):
    pos = pos[:,:2]
    N = pos.shape[0]

    inds = np.sort(list(c))
        
    if plt_objs is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt_objs

    if usetex:
        plt_utils.setup_tex()

    rho = np.sum(M[:,np.unique(inds)]**2,axis=1)
    sizes = np.ones(N) * dotsize
    big_rho = rho > big_rho_threshold
    sizes[big_rho] *= rel_center_size
    colors=rho
    # colors[big_rho] = np.log(1.5+rho[big_rho])
    ye = ax.scatter(pos.T[0], pos.T[1], c=colors, s=sizes, cmap='plasma',zorder=2,vmin=vmin,vmax=vmax)
    cbar = fig.colorbar(ye,ax=ax,orientation='vertical')

    # This should really be its own function(s)... :p
    if show_bonds:
        # Filter positions to avoid plotting unnecessary bonds (can be a lot)
        if xbounds is not None:
            xmin, xmax = xbounds
            xfilter = (pos[:,0] >= xmin) * (pos[:,0] <= xmax)
        else:
            yfilter = np.ones(pos.shape[0],dtype=bool)
        if ybounds is not None:
            ymin, ymax = ybounds
            yfilter = (pos[:,1] >= ymin) * (pos[:,1] <= ymax)
        else:
            yfilter = np.ones(pos.shape[0],dtype=bool)
        pos_filter = xfilter * yfilter
        filtered_pos = pos[pos_filter]
        filtered_rho = rho[pos_filter]
 
        # Build adjmat
        rCC = 1.8
        A = adjacency_matrix_sparse(filtered_pos,rCC)
        pairs = np.vstack(A.nonzero()).T
        
        # Define colormap for bonds
        if vmin is not None or vmax is not None:
            norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax) 
            cmap = cm.plasma
            float2clr_map = cm.ScalarMappable(norm=norm,cmap=cmap)

        # Plot bonds
        for ij in pairs:
            i,j = ij
            rhoi = filtered_rho[i]
            rhoj = filtered_rho[j]
            ax.plot([filtered_pos[i,0],filtered_pos[j,0]],[filtered_pos[i,1],filtered_pos[j,1]],color=float2clr_map.to_rgba(np.max([rhoi,rhoj])),ls='-',lw=3.0,zorder=1)


    if xbounds is not None:
        ax.set_xlim(xbounds)

    if ybounds is not None:
        ax.set_ylim(ybounds)

    ax.set_aspect('equal')
    ax.set_xlabel("$x$ [\AA]")
    ax.set_ylabel("$y$ [\AA]")

    

    if show:
        plt.show()


def plot_loc_centers(rho, xedges, yedges, centers, colours='r', show=True, plt_objs=None):

    if plt_objs is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt_objs
    
    ax.imshow(rho, origin='lower',extent=[*xedges[[0,-1]], *yedges[[0,-1]]],zorder=0)
    ax.scatter(*centers.T,c=colours,marker='*',s=5.0,zorder=2)
    
    if show:
        plt.show()
    else:
        return fig, ax


def dcrit_hists(dcrits,temps,nbins,plot_inds=None,colormap='coolwarm',alpha=0.6,usetex=True,plt_objs=None,show=True,fontsize=20):
    Tcm = plt_utils.get_cm(temps,colormap,max_val=1.0)
    
    # If indices aren't specified, plot all the histograms
    # We still pass all of the dcrits to this function (even those we don't plot) to get a better
    # contrasted colormap.
    if plot_inds is None:
        plot_inds = range(len(temps))


    if usetex:
        plt_utils.setup_tex(fontsize=fontsize)
    
    if plt_objs is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt_objs

    for k in plot_inds:
        plt_utils.histogram(dcrits[:,k],nbins=nbins,show=False, density=True, plt_objs=(fig,ax),
            plt_kwargs={'alpha': alpha, 'color': Tcm[k], 'label': f'$T = {temps[k]}$K'})
    
    ax.set_xlabel('Critical distance $\\xi_{c}$')
    ax.set_ylabel('$P(\\xi_c)$')

    if show:
        plt.legend(fontsize=fontsize)
        plt.show()
    else:
        return fig, ax
