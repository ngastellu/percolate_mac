#!/usr/bin/env python

from itertools import combinations, starmap
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import find_peaks
from numba import njit
import qcnico.qchemMAC as qcm
from qcnico.graph_tools import components
from qcnico import plt_utils

@njit
def energy_distance(ei, ej, mu, T):
    kB = 8.617e-5
    return np.exp( (np.abs(ei-mu) + np.abs(ej-mu) + np.abs(ej-ei)) / (2*kB*T) )

@njit
def miller_abrahams_distance(ei, ej, ri, rj, mu, T, a0):
    kB = 8.617e-5 #eV/K
    return np.exp( ((np.abs(ei-mu) + np.abs(ej-mu) + np.abs(ej-ei)) / (2*kB*T)) + 2*np.linalg.norm(ri - rj)/a0 )

@njit
def log_miller_abrahams_distance(ei, ej, ri, rj, mu, T, a0):
    kB = 8.617e-5 #eV/K
    return ((np.abs(ei-mu) + np.abs(ej-mu) + np.abs(ej-ei)) / (2*kB*T)) + 2*np.linalg.norm(ri - rj)/a0

@njit
def diff_arrs(e, coms, a0, eF=0):
    N = e.shape[0]
    ddarr = np.zeros(int(N*(N-1)/2))
    edarr = np.zeros(int(N*(N-1)/2))
    k = 0
    for i in range(N):
        for j in range(i):
            edarr[k] = np.abs(e[i]-eF) + np.abs(e[j]-eF) + np.abs(e[i] - e[j])
            ddarr[k] = 2*np.linalg.norm(coms[i]-coms[j])/a0
            k += 1
    return edarr, ddarr



@njit
def dArray_energy(e, T, eF=None):
    N = e.size
    if eF is None:
        eF = 0.5 * (e[N//2 - 1] + e[N//2])
    darr = np.zeros(N*(N-1)//2)
    k = 0

    for i in range(1,N):
        for j in range(i):
            darr[k] = energy_distance(e[i], e[j], eF, T)
            k += 1
    
    return darr

@njit
def dArray_MA(e, coms, T, a0=1, eF=None):
    N = e.size
    if eF is None:
        eF = 0.5 * (e[N//2 - 1] + e[N//2])
    darr = np.zeros(N*(N-1)//2)
    k = 0

    for i in range(1,N):
        for j in range(i):
            darr[k] = miller_abrahams_distance(e[i], e[j], coms[i], coms[j], eF, T, a0)
            k += 1

    return darr

@njit
def dArray_logMA(e, coms, T, a0=1, eF=None):
    N = e.size
    if eF is None:
        eF = 0.5 * (e[N//2 - 1] + e[N//2])
    darr = np.zeros(N*(N-1)//2)
    k = 0

    for i in range(1,N):
        for j in range(i):
            darr[k] = log_miller_abrahams_distance(e[i], e[j], coms[i], coms[j], eF, T, a0)
            k += 1

    return darr

def distance_array_itertools(e, T):
    N = e.size
    print(N)
    eF = 0.5 * (e[N//2 - 1] + e[N//2])
    
    e_pairs = combinations(e,2)
    return np.array(list(starmap(partial(energy_distance, mu=eF, T=T), e_pairs)))

@njit
def pair_inds(n,N):
    zero_k_inds = np.array([k*(k-1)//2 for k in range(1,N)])
    i_inds = np.array([np.sum(nn >= zero_k_inds) for nn in n])
    return i_inds, (n - zero_k_inds[i_inds-1])

def k_ind(i,j): return int(i*(i-1)/2 + j)

def LR_sites(pos, M, gamma, site_inds, tolscal=3.0):
    
    # Get MO couplings
    agaL, agaR = qcm.AO_gammas(pos, gamma)
    gamL, gamR = qcm.MO_gammas(M,agaL, agaR, return_diag=True)

    # Define high-coupling threshold
    gamL_tol = np.mean(gamL) + tolscal*np.std(gamL)
    gamR_tol = np.mean(gamR) + tolscal*np.std(gamR)

    # 'Transform' from MO labeling to site labeling
    sgamL = gamL[site_inds]
    sgamR = gamR[site_inds]

    # Get strongly coupled sites
    L = set(sgamL >= gamL_tol)
    R = set(sgamR >= gamR_tol)

    return L, R

def bin_centers(peak_inds,xedges,yedges):
    centers = np.zeros((len(peak_inds),2))
    for k, ij in enumerate(peak_inds):
        i,j = ij
        x = 0.5*(xedges[i]+xedges[i+1])
        y = 0.5*(yedges[j]+yedges[j+1])
        centers[k,:] = [x,y]
    return centers


def get_MO_loc_centers(pos, M, n, nbins=20, threshold_ratio=0.60,return_realspace=True,padded_rho=True,return_gridify=False):
    rho, xedges, yedges = qcm.gridifyMO(pos, M, n, nbins,True)
    if padded_rho:
        nbins = nbins+2 #nbins describes over how many bins the actual MO is discretized; doesn't account for padding
    
    all_peaks = {}
    for i in range(1,nbins-1):
        data = rho[i,:]
        peak_inds, _ = find_peaks(data)
        for j in peak_inds:
            peak_val = data[j]
            if peak_val > 1e-4: all_peaks[(i,j)] = peak_val

    threshold = max(all_peaks.values())*threshold_ratio
    peaks = {key:val for key,val in all_peaks.items() if val >= threshold}

    # Some peaks still occupy several neighbouring pixels; keep only the most prominent pixel
    # so that we have 1 peak <---> 1 pixel.
    pk_inds = set(peaks.keys())
    shift = np.array([[0,1],[1,0],[1,1],[0,-1],[-1,0],[-1,-1],[1,-1],[-1,1]])
    
    while pk_inds:
        ij = pk_inds.pop()
        nns = set(tuple(nm) for nm in ij + shift)
        intersect = nns & pk_inds
        for nm in intersect:
            if peaks[nm] <= peaks[ij]:
                #print(nm, peaks[nm])
                peaks[nm] = 0
            else:
                peaks[ij] = 0

    #need to swap indices of peak position; 1st index actually labels y and 2nd labels x
    peak_inds = np.roll([key for key in peaks.keys() if peaks[key] > 0],shift=1,axis=1)
    #peak_inds = np.array([key for key in peaks.keys() if peaks[key] > 0])
    if return_realspace and return_gridify:
        return bin_centers(peak_inds,xedges,yedges), rho, xedges, yedges
    elif return_realspace and (not return_gridify):
        return bin_centers(peak_inds,xedges,yedges)
    elif return_gridify and (not return_realspace):
        return peak_inds, rho, xedges, yedges
    else:
        return peak_inds
    

def correct_peaks(sites, pos, rho, xedges, yedges, side):
    x = pos[:,0]
    length = np.max(x) - np.min(x)
    midx = length/2

    if side == 'L':
        goodbools = sites[:,0] < midx
    else: # <==> side == 'R'
        goodbools = sites[:,0] > midx
    
    # First, remove bad sites
    sites = sites[goodbools]

    # Check if any sites are left, if not, add peak on the right edge, at the pixel with the highest density
    if not np.any(goodbools):
        print('!!! Generating new peaks !!!')
        if side == 'L':
            edge_ind = 1
        else: # <==> side == 'R' 
            edge_ind = -3
        peak_ind = np.argmax(rho[:,edge_ind]) -1 

        sites = bin_centers([(edge_ind,peak_ind)],xedges,yedges)
        print(sites)
    
    return sites

def generate_site_list(pos,M,L,R,energies,nbins=20,threshold_ratio=0.60):
    centres = np.zeros(2) #setting centers = [0,0] allows us to use np.vstack when constructing centres array
    ee = []
    inds = []
    for n in range(M.shape[1]):
        cc, rho, xedges, yedges = get_MO_loc_centers(pos,M,n,nbins,threshold_ratio,return_gridify=True)
        if n in L:
            print(n)
            cc = correct_peaks(cc, pos, rho, xedges, yedges,'L')
        
        elif n in R:
            print(n)
            cc = correct_peaks(cc, pos, rho, xedges, yedges,'R')
        

        centres = np.vstack([centres,cc])
        ee.extend([energies[n]]*cc.shape[0])
        inds.extend([n]*cc.shape[0]) #this will help us keep track of which centers belong to which MOs
    return centres[1:,:], np.array(ee), np.array(inds) #get rid of initial [0,0] entry in centres

    
def percolate(e, pos, M, T=300, a0=1, eF=None, dArrs=None, 
                gamL_tol=0.07,gamR_tol=0.07,gamma=0.1, MOgams=None, coupled_MO_sets=None,
                distance='miller_abrahams', 
                return_adjmat=False):
    
    assert distance in ['energy', 'miller_abrahams', 'logMA'], 'Invalid distance argument. Must be either "miller-abrahams" (default) or "energy".'
    if distance == 'energy':
        darr = dArray_energy(e,T, eF)
    elif distance == 'miller_abrahams':
        MO_coms = qcm.MO_com(pos, M)
        darr = dArray_MA(e, MO_coms, T, a0, eF)
    elif distance == 'logMA' and dArrs is None:
        MO_coms = qcm.MO_com(pos,M)
        darr = dArray_logMA(e, MO_coms, T, a0, eF)
    else:
        kB = 8.617e-5
        edarr, ddarr = dArrs
        darr = ddarr  + (edarr / (kB*T))
    # np.save('darr.npy', darr)
    if MOgams is None:
        agaL, agaR = qcm.AO_gammas(pos,gamma)
        gamLs, gamRs = qcm.MO_gammas(M, agaL, agaR, return_diag=True)
    else:
        gamLs, gamRs = MOgams
    if coupled_MO_sets is None:
        L = set((gamLs > gamL_tol).nonzero()[0])
        R = set((gamRs > gamR_tol).nonzero()[0])
    else:
        L, R = coupled_MO_sets
    N = e.size

    percolated = False
    darr_sorted = np.sort(darr)
    adj_mat = np.zeros((N,N),dtype=bool)
    spanning_clusters = []
    d_ind = 0
    while (not percolated) and (d_ind < N*(N-1)//2):
        d = darr_sorted[d_ind] #start with smallest distance and move up                                                                                                                                              
        print('d = ', d)       
        connected_inds = (darr < d).nonzero()[0] #darr is 1D array     
        print('Nb. of connected pairs = ', len(connected_inds))
        ij = pair_inds(connected_inds,N)
        print(ij)
        adj_mat[ij] = True
        adj_mat  += adj_mat.T
        print(adj_mat.astype(int))
        print('\n')
        relevant_MOs = set(np.unique(ij))
        coupledL = not L.isdisjoint(relevant_MOs)
        coupledR = not R.isdisjoint(relevant_MOs)
        if coupledL and coupledR:
            print('Getting clusters...')
            clusters = components(adj_mat)
            print('Done! Now looping over clusters...')
            print(f'Nb. of clusters with more than 1 MO = {np.sum(np.array([len(c) for c in clusters])>1)}')
            for c in clusters:
                #print('Size of cluster: ', len(c))
                #print('Cluster: ', c)
                if (not c.isdisjoint(L)) and (not c.isdisjoint(R)):
                    spanning_clusters.append(c)
                    percolated = True
                    # if first_try:
                    #     print('First try!')
            print('Done with clusters loop!')
        
        d_ind += 1
        # d = darr_sorted[d_ind]
        # first_try = False
    
    if d_ind == (N*(N-1)//2)-1:
        print(f'No spanning clusters found at T = {T}K.')
        return clusters, d, adj_mat
    
    if return_adjmat:
        return spanning_clusters, d, adj_mat
    else:
        return spanning_clusters, d

    
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


def plot_cluster_brute_force(c,pos, M, adjmat,show_densities=False,dotsize=20, usetex=True, show=True, centers=None, rel_center_size=2.0, inds=None, plt_objs=None):
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

    # Draw sites
    ax.scatter(*centers.T, marker='*', c='r', s = rel_center_size*dotsize,zorder=2)
    ax.set_aspect('equal')
    ax.set_xlabel("$x$ [\AA]")
    ax.set_ylabel("$y$ [\AA]")

    # Draw edges between each site and its neighbours
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


def plot_loc_centers(rho, xedges, yedges, centers, colours='r', show=True, plt_objs=None):

    if plt_objs is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt_objs
    
    ax.imshow(rho, origin='lower',extent=[*xedges[[0,-1]], *yedges[[0,-1]]])
    ax.scatter(*centers.T,c=colours,marker='*',s=5.0)
    if show:
        plt.show()




if __name__ == '__main__':
    from os import path
    import sys
    import pickle
    from qcnico.qcffpi_io import read_energies, read_MO_file

    qcffpidir = path.expanduser('~/scratch/MO_airebo_dynamics/qcffpi')
    mo_file = 'MO_coefs.dat'
    energy_file = 'orb_energy.dat'
    
    n1 = int(sys.argv[1])
    n2 = int(sys.argv[2])
    step = int(sys.argv[3])

    frames = np.arange(n1,n2,step)
    nframes = frames.size

    cluster_list = [None] * nframes
    ds = np.zeros((nframes,2))
    for k, i in range(frames):
        framedir= f'300K_initplanar_norotate/frame-{i}'
        mo_path = path.join(qcffpidir,framedir,mo_file)
        energy_path = path.join(qcffpidir,framedir,energy_file)

        energies = read_energies(energy_path)
        pos, M = read_MO_file(mo_path)

        clusters, d = percolate(energies,pos,M)

        cluster_list[k] = clusters
        ds[k] = i, d
    
    np.save(ds, f'ds_frames_{n1}-{n2}-{step}.npy')
    with open('clusters_frames_{n1}-{n2}-{step}.pkl', 'wb') as fo:
        pickle.dump(cluster_list, fo)
    