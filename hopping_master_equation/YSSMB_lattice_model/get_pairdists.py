#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
# from scalene import scalene_profiler
 
# @profile
@njit
def get_pairdists(pos, dists):
    """This function's job is to loop over all pairs of distinct lattice sites to compute:
    * the pairwise distances between all lattice sites
    """
    print("In Numba land.")
    N = pos.shape[0]
    k = 0
    for i in range(N):
        for j in range(i):
            dists[k] = np.linalg.norm(pos[j,:] - pos[i,:])
            print(k)
            k += 1
    return dists



# -------------- MAIN --------------




N1 = 32
N2 = 16

N = N1 * N2 * N2

# scalene_profiler.start()

dists = np.zeros(N*(N-1)//2)

pos = np.load(f'/Users/nico/Desktop/simulation_outputs/yssmb_hopping/lattice_{N1}x{N2}x{N2}.npy')

dists = get_pairdists(pos, dists)

# scalene_profiler.end()

np.save(f'pairdists_{N1}x{N2}x{N2}.npy', dists)

# np.save(f'energy_correlations/ecorr-{n}.npy', corr)

