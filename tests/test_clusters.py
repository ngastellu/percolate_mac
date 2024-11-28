#!/usr/bin/env python

import numpy as np
from qcnico.graph_tools import components
from percolate import pair_inds


N = 3534
darr = np.load('darr.npy')
cinds = (darr < 10).nonzero()[0]
ij = pair_inds(cinds, N)

M = np.zeros((N,N))

M[ij] = 1
M += M.T

clusters = components(M)

lls = np.array([len(c) for c in clusters])
ye = (lls > 1).nonzero()[0]
print('Number of clusters with more than one MO = ', len(ye))

rcs = [clusters[i] for i in ye]
bb = np.array([c == rcs[0] for c in rcs])
print(np.all(bb))
