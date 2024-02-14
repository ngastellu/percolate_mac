#!/usr/bin/env python

import numpy as np
from qcnico.coords_io import read_xsf
from qcnico.qcplots import plot_MO
from qcnico.remove_dangling_carbons import remove_dangling_carbons



MOfile = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/MOs_ARPACK/virtual/MOs_ARPACK_bigMAC-10.npy'
boyzfile = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/boyz_ARPACK/virtual/boyz_MOs_bigMAC-10.npy' 

strucfile = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/structures/bigMAC-10_relaxed.xsf'
pos, _ = read_xsf(strucfile)
pos = remove_dangling_carbons(pos,1.8)
print(pos.shape)

M = np.load(MOfile)
Mboyz = np.load(boyzfile)

nMOs = M.shape[1]

print(M.shape)
print(Mboyz.shape)


sample_inds = np.random.randint(nMOs,size=5)

for n in sample_inds:
    plot_MO(pos, M, n, dotsize=5.0)
    plot_MO(pos, Mboyz, n, dotsize=5.0, cmap='viridis')