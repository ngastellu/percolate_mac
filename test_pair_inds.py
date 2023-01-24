#!/usr/bin/env python

import numpy as np
from numpy.random import default_rng
from itertools import combinations
from percolate import pair_inds

rng = default_rng(64)

N = 5
M = N * (N-1) // 2 #number of i,j pairs such that j < i <= N
ij0 = np.array(list(combinations(range(N),2)))
print(ij0.shape)

#ye = rng.integers(M,size=M)
ye = np.array([k**2 for k in range(1,M+1)])

arr = np.zeros((N,N))

arr[ij0[:,0], ij0[:,1]] = ye
arr = arr.T
print(arr)

