#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
# from scalene import scalene_profiler
 
# @profile
@njit
def get_pair_qties(n, pos, energies, corr, dists):
    """This function's job is to loop over all pairs of distinct lattice sites to compute:
    * the pairwise distances between all lattice sites
    * the energy correlation between lattice sites 
    """
    print("In Numba land.")
    N = energies.shape[0]
    k = 0
    for i in range(N):
        for j in range(i):
            if n == 1:
                dists[k] = np.abs(pos[j] - pos[i])
            corr[k] += energies[j] * energies[i]
            print(k)
            k += 1
    return dists, corr



# -------------- MAIN --------------


n = int(sys.argv[1])

# N1 = 64
# N2 = 32

N = int(sys.argv[2])
# scalene_profiler.start()

corr = np.zeros(N*(N-1)//2)
dists = np.zeros(N*(N-1)//2)

pos = np.hstack((np.arange(-int(N/2),0), np.arange(1,int(N/2)+1)))

energies = np.load(f'corr_energies_1d_{N}/correlated_energies-{n}.npy').ravel()
dists, corr = get_pair_qties(n,pos,energies, corr, dists)

# scalene_profiler.end()

if n == 0:
    np.save(f'pairdists_{N}.npy', dists)

if not os.path.exists(f'energy_correlations_{N}'):
    os.mkdir(f'energy_correlations_{N}')

np.save(f'energy_correlations_{N}/ecorr-{n}.npy', corr)

