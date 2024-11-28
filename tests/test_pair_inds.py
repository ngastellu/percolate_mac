#!/usr/bin/env python

import numpy as np
from numpy.random import default_rng
from itertools import combinations
from percolate import pair_inds

rng = default_rng(64)

N = 100
M = N * (N-1) // 2 #number of i,j pairs such that j < i <= N
#ij0 = np.array(list(combinations(range(N),2)))
#print(ij0.shape)

#ye = rng.integers(M,size=M)
arr1d = rng.permutation(np.array([k**2 for k in range(1,M+1)]))


arr2d = np.zeros((N,N),dtype=int)

#arr2d[ij0[:,0], ij0[:,1]] = arr1d
#arr2d = arr2d.T

k = 0
for i in range(N):
    for j in range(i):
        arr2d[i][j] = arr1d[k]
        k+=1

print(arr1d)
print(arr2d)

ij = np.vstack(arr2d.nonzero()).T


#n = rng.integers(M,size=7)
n = np.arange(M)
print('Selected inds from 1D array: ', n)
ij1 = pair_inds(n, N)
print('Corresponding inds from 2D array: ', np.vstack(ij1).T)
#print('Inds used to populate 2D array: ', ij0)


print('Elements of 1D array corresponding to above inds: ', arr1d[n])
print('Elements of 2D array corresponding to above inds: ', arr2d[ij1])

print(np.all(arr1d[n] == arr2d[ij1]))




