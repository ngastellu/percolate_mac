#!/usr/bin/env python

import numpy as np
from percolate import avg_nb_neighbours


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays) #use type promotion to match types of all input arrays
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)): #ix_ magic, see my notes on ix_ hacks
        arr[...,i] = a
    return arr.reshape(-1, la)


energies = np.zeros(16)
energies[:4] = 0
energies[4:8] = 1
energies[8:12] = 2
energies[12:16] = 3

pos = cartesian_product(np.arange(4),np.arange(4))
print(pos)

dists = np.linalg.norm(pos[:,None,:] - pos,axis=2)
for i in range(16):
    dists[i,i] = 1000


erange = np.linspace(0,4,9)
drange = np.arange(6)

B = avg_nb_neighbours(energies, dists, erange, drange)
print(B)

print(np.all(B == B[0,:]))