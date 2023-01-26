#!/usr/bin/env python

from itertools import combinations, starmap
from functools import partial
import numpy as np
from scipy.spatial import cKDTree
from numba import njit
import qcnico.qchemMAC as qcm
from qcnico.graph_tools import components

@njit
def energy_distance(ei, ej, mu):
    return np.abs(ei-mu) + np.abs(ej-mu) + np.abs(ej-ei)

@njit
def distance_array(e):
    N = e.size
    eF = 0.5 * (e[N//2 - 1] + e[N//2])
    darr = np.zeros(N*(N-1)//2)
    k = 0

    for i in range(1,N):
        for j in range(i):
            darr[k] = energy_distance(e[i], e[j], eF)
            k += 1
    
    return darr

def distance_array_itertools(e):
    N = e.size
    print(N)
    eF = 0.5 * (e[N//2 - 1] + e[N//2])
    
    e_pairs = combinations(e,2)
    return np.array(list(starmap(partial(energy_distance, mu=eF), e_pairs)))

@njit
def pair_inds(n,N):
    zero_k_inds = np.array([k*(k-1)//2 for k in range(1,N)])
    i_inds = np.array([np.sum(nn >= zero_k_inds) for nn in n])
    return i_inds, (n - zero_k_inds[i_inds-1])
    
def percolate(e, pos, M, dmin=0, dstep=1e-3, gamma_tol=0.07, gamma=0.1):
    darr = distance_array(energies)
    N = e.size
    percolated = False
    d = dmin
    adj_mat = np.zeros((N,N))
    agaL, agaR = qcm.AO_gammas(pos,gamma)
    gamLs, gamRs = qcm.MO_gammas(M, agaL, agaR, return_diag=True)
    L = set((gamLs > gamma_tol).nonzero()[0])
    R = set((gamRs > gamma_tol).nonzero()[0])
    print(len(L))
    print(len(R))
    spanning_clusters = []
    while not percolated:                                                                                                                                                  
        print(d)
        connected_inds = (darr < d).nonzero()[0]
        ij = pair_inds(connected_inds,N)
        adj_mat[ij] = 1
        adj_mat  = (adj_mat + adj_mat.T) // 2
        relevant_MOs = set(np.unique(ij))
        coupledL = not L.isdisjoint(relevant_MOs)
        coupledR = not R.isdisjoint(relevant_MOs)
        if np.any(coupledL) and np.any(coupledR) > gamma_tol:
            print('Getting clusters...')
            clusters = components(adj_mat)
            print('Done!')
            for c in clusters:
                if (not c.disjoint(L)) and (not c.disjoint(R)):
                    spanning_clusters.append(c)
                    percolated = True
        
        d += dstep
    
    return spanning_clusters, d

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