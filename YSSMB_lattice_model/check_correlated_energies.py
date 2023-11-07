#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
# from qcnico.graph_tools import pairwise_dists
from qcnico.plt_utils import setup_tex
from numba import njit

@njit
def pair_inds(n,N):
    zero_k_inds = np.array([k*(k-1)//2 for k in range(1,N)])
    i_inds = np.array([np.sum(nn >= zero_k_inds) for nn in n])
    return i_inds, (n - zero_k_inds[i_inds-1])

def k_ind(i,j): return int(i*(i-1)/2 + j)


def data_generator(nsamples):
    """This generator allows me to pass the energies stored inside of NPY files to my Numba-accelerated 
    function `get_pair_qties` without using `np.load` inside of it (Numba doesn't allow reading/writing 
    to disk) and without having to pre-load all of the energies and store them in a big array (not
    memory efficient)."""
    for n in range(nsamples):
        yield np.load(f'corr_energies/correlated_energies-{n+1}.npy')


def compute_correlations(nsamples):
    """Pure Python function that loops over disorder realisations, loads site energies, and calls the 
    Numba-accelerated function that does the actual computation."""
    N1 = 64
    N2 = 32

    N = N1 * N2 * N2

    corr = np.zeros(N*(N-1)//2)
    dists = np.zeros(N*(N-1)//2)

    pos = np.load('lattice.npy')

    for n in range(nsamples):
        print(n)
        energies = np.load(f'corr_energies/correlated_energies-{n+1}.npy').ravel()
        dists, corr = get_pair_qties(n,pos,energies, corr, dists)
    
    corr /= nsamples
    return dists, corr
    

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
nsamples = 200
# pos = np.load('lattice.npy')
# energen = data_generator(nsamples)
# dists, corr = get_pair_qties(pos, energen)

dists, corr = compute_correlations(nsamples)

setup_tex()

plt.plot(dists,corr)
plt.xlabel('$R_{ij}$ [\AA]')
plt.ylabel('$\langle\\varepsilon_i\\varepsilon_j\\rangle$')
plt.show()





# for n in range(nsamples):
#     print(n)
#     energies = np.load(f'corr_energies/correlated_energies-{n+1}.npy')
#     print('Loaded.')
#     energies = energies.reshape(N)
#     print('Reshaped.')
#     corr += (energies[None,:] * energies[:,None])





# corr = np.ravel(corr/nsamples)
# lattice = np.load('lattice.npy')
# dists = np.ravel(pairwise_dists(lattice))
