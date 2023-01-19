#!/usr/bin/env python

from itertools import combinations, starmap
from functools import partial
import numpy as np
from scipy.spatial import cKDTree
from numba import njit

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

@nijt
def pair_inds(n,N):
    pass
    

def percolate(e, pos, M, dmin=0, dstep=1e-3):
    darr = distance_array(energies)
    N = e.size
    percolated = False
    d = dmin
    adj_mat = np.zeros((N,N))
    while not percolated:
        pass

    



if __name__ == '__main__':
    from os import path
    from time import perf_counter
    from qcnico.qcffpi_io import read_energies, read_MO_file

    datadir = path.expanduser('~/Desktop/simulation_outputs/qcffpi_data/MO_dynamics/300K_initplanar_norotate/orbital_energies')
    datafile = 'orb_energy_frame-80000.dat'
    datapath = path.join(datadir,datafile)

    energies = read_energies(datapath)