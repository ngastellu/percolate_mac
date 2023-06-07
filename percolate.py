#!/usr/bin/env python

from itertools import combinations, starmap
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
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
def dArray_energy(e, T):
    N = e.size
    eF = 0.5 * (e[N//2 - 1] + e[N//2])
    darr = np.zeros(N*(N-1)//2)
    k = 0

    for i in range(1,N):
        for j in range(i):
            darr[k] = energy_distance(e[i], e[j], eF, T)
            k += 1
    
    return darr

@njit
def dArray_MA(e, coms, T, a0=1):
    N = e.size
    eF = 0.5 * (e[N//2 - 1] + e[N//2])
    darr = np.zeros(N*(N-1)//2)
    k = 0

    for i in range(1,N):
        for j in range(i):
            darr[k] = miller_abrahams_distance(e[i], e[j], coms[i], coms[j], eF, T, a0)
            k += 1

    return darr

@njit
def dArray_logMA(e, coms, T, a0=1):
    N = e.size
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
    
def percolate(e, pos, M, dmin=0, dstep=1e-3, gamL_tol=0.07,gamR_tol=0.07,gamma=0.1, T=300, distance='miller_abrahams', return_adjmat=False):
    assert distance in ['energy', 'miller_abrahams', 'logMA'], 'Invalid distance argument. Must be either "miller-abrahams" (default) or "energy".'
    if distance == 'energy':
        darr = dArray_energy(e,T)
    elif distance == 'miller_abrahams':
        MO_coms = qcm.MO_com(pos, M)
        darr = dArray_MA(e, MO_coms, T)
    else:
        MO_coms = qcm.MO_com(pos,M)
        darr = dArray_logMA(e, MO_coms, T)
    np.save('darr.npy', darr)
    N = e.size
    percolated = False
    d = dmin
    adj_mat = np.zeros((N,N))
    agaL, agaR = qcm.AO_gammas(pos,gamma)
    gamLs, gamRs = qcm.MO_gammas(M, agaL, agaR, return_diag=True)
    L = set((gamLs > gamL_tol).nonzero()[0])
    R = set((gamRs > gamR_tol).nonzero()[0])
    spanning_clusters = []
    while not percolated:                                                                                                                                              
        print('d = ', d)            
        connected_inds = (darr < d).nonzero()[0]
        print('Nb. of connected pairs = ', len(connected_inds))
        ij = pair_inds(connected_inds,N)
        adj_mat[ij] = 1
        adj_mat  += adj_mat.T
        relevant_MOs = set(np.unique(ij))
        coupledL = not L.isdisjoint(relevant_MOs)
        coupledR = not R.isdisjoint(relevant_MOs)
        if np.any(coupledL) and np.any(coupledR) > gamma_tol:
            print('Getting clusters...')
            clusters = components(adj_mat)
            print('Done! Now looping over clusters...')
            print(f'Nb. of clusters with more MO = {np.sum(np.array([len(c) for c in clusters])>1)}')
            for c in clusters:
                #print('Size of cluster: ', len(c))
                #print('Cluster: ', c)
                if (not c.isdisjoint(L)) and (not c.isdisjoint(R)):
                    spanning_clusters.append(c)
                    percolated = True
            print('Done with clusters loop!')
        
        d += dstep
    
    if return_adjmat:
        return spanning_clusters, d, adj_mat
    else:
        return spanning_clusters, d
    

def plot_cluster(c,pos, M, adjmat,show_densities=False, dotsize=20, usetex=True, show=True):
    pos = pos[:,:2]
    c = np.sort(list(c))
    centers = qcm.MO_com(pos,M,c)

    fig, ax = plt.subplots()

    if usetex:
        plt_utils.setup_tex()

    if show_densities:
        rho = np.sum(M[:,c]**2,axis=1)
        ye = ax.scatter(pos.T[0], pos.T[1], c=rho, s=dotsize, cmap='plasma',zorder=1)
        cbar = fig.colorbar(ye,ax=ax,orientation='vertical')

    else:
        ax.scatter(pos.T[0], pos.T[1], c='k', s=dotsize)

    ax.scatter(*centers.T, marker='*', c='r', s = 1.2*dotsize,zorder=2)
    seen = set()
    for i in c:
        if i not in seen:
            n = np.sum(i > c)
            r1 = centers[n]
            neighbours = adjmat[i,:].nonzero()[0]
            seen.update(neighbours)
            for j in neighbours:
                m = np.sum(j > c)
                r2 = centers[m]
                pts = np.vstack((r1,r2)).T
                ax.plot(*pts, 'r-', lw=0.7)
    
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
    