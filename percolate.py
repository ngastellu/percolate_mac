#!/usr/bin/env python

from itertools import combinations, starmap
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from numba import njit
import qcnico.qchemMAC as qcm
from qcnico.graph_tools import components

@njit
def energy_distance(ei, ej, mu, T):
    kB = 8.617e-5
    return np.exp( (np.abs(ei-mu) + np.abs(ej-mu) + np.abs(ej-ei)) / (2*kB*T) )

@njit
def distance_array(e, T):
    N = e.size
    eF = 0.5 * (e[N//2 - 1] + e[N//2])
    darr = np.zeros(N*(N-1)//2)
    k = 0

    for i in range(1,N):
        for j in range(i):
            darr[k] = energy_distance(e[i], e[j], eF, T)
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
    
def percolate(e, pos, M, dmin=0, dstep=1e-3, gamma_tol=0.07, gamma=0.1, T=300, return_adjmat=False):
    darr = distance_array(e,T)
    np.save('darr.npy', darr)
    N = e.size
    percolated = False
    d = dmin
    adj_mat = np.zeros((N,N))
    agaL, agaR = qcm.AO_gammas(pos,gamma)
    gamLs, gamRs = qcm.MO_gammas(M, agaL, agaR, return_diag=True)
    L = set((gamLs > gamma_tol).nonzero()[0])
    R = set((gamRs > gamma_tol).nonzero()[0])
    spanning_clusters = []
    while not percolated:                                                                                                                                              
        print(d)            
        connected_inds = (darr < d).nonzero()[0]
        print(len(connected_inds))
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


def plot_cluster(c,pos, M, adjmat, cmap='inferno',show_densities=False, usetex=True):
    if isinstance(c,set): c = list(c)
    centers = qcm.MO_com(pos,M,c)

    fig, ax = plt.subplots()

    if show_densities:
        rho = np.sum(M[:,c]**2,axis=1)
        ye = ax.scatter(pos.T[0], pos.T[1], c=rho, s=dotsize, cmap='plasma')

    else:
        ye = ax.scatter(pos.T[0], pos.T[1], c='k', s=dotsize)

    ax.scatter(*centers.T, marker='*', c='r', s = 1.2*dotsize)
    seen = set()
    for n in c:
        if n not in seen:
            neighbours = adjmat[n,:].nonzero()[0]
            seen.update(neighbours)
            for m in neighbours:
                ax.plot()






if __name__ == '__main__':
    from os import path
    from time import perf_counter
    from qcnico.qcffpi_io import read_energies, read_MO_file

    datadir = path.expanduser('~/Desktop/simulation_outputs/qcffpi_data/MO_dynamics/300K_initplanar_norotate/')
    energy_dir = 'orbital_energies'
    mo_dir = 'MO_coefs'

    frame_nb = 80000
    mo_file = f'MOs_frame-{frame_nb}.dat'
    energy_file = f'orb_energy_frame-{frame_nb}.dat'
    
    mo_path = path.join(datadir,mo_dir,mo_file)
    energy_path = path.join(datadir,energy_dir,energy_file)

    energies = read_energies(energy_path)
    pos, M = read_MO_file(mo_path)

    clusters, d = percolate(energies,pos,M)