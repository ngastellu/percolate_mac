#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
 

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
            if n == 0:
                dists[k] = np.linalg.norm(pos[j,:] - pos[i,:])
            corr[k] += energies[j] * energies[i]
            print(k)
            k += 1
    return dists, corr



# -------------- MAIN --------------
n = int(sys.argv[1])

N1 = 64
N2 = 32

N = N1 * N2 * N2

corr = np.zeros(N*(N-1)//2)
dists = np.zeros(N*(N-1)//2)

pos = np.load('lattice.npy')

energies = np.load(f'corr_energies/correlated_energies-{n+1}.npy').ravel()
dists, corr = get_pair_qties(n,pos,energies, corr, dists)

if n == 0:
    np.save('pairdists.npy', dists)

np.save(f'energy_correlations/ecorr-{n}.npy', corr)

