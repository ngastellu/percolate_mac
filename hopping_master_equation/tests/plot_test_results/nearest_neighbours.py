#!/usr/bin/env python

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt 


rng = default_rng(0)

pos2 = np.load("../full_device_test/2d/init/pos.npy")
nn2 = np.load("../full_device_test/2d/init/nn2.npy")
edge_size = 20
N = pos2.shape[0]
print(N)


sample_inds = rng.integers(2*edge_size,N-2*edge_size,size=10)
sample_inds = np.hstack(([N-2*edge_size-4],sample_inds))

for i in sample_inds[:5]:
    plt.scatter(*pos2[2*edge_size:N-2*edge_size].T,c='k')
    plt.scatter(*pos2[:2*edge_size].T,c='gray')
    plt.scatter(*pos2[N-2*edge_size:].T,c='gray')
    plt.scatter(pos2[i,0],pos2[i,1],c='r')
    neighbours = nn2[i,:]
    neighbours = neighbours[neighbours > 0] - 1
    print(f"{i} ---> {neighbours}")
    print(f"{pos2[i,:]}")
    for j in neighbours:
        print(f"{j} ---> {pos2[j,:]}")
        if j >= edge_size and j < N-edge_size:
            plt.scatter(pos2[j,0],pos2[j,1],c='b')
        else:
            print('yeyeye')
            plt.scatter(pos2[j,0],pos2[j,1],facecolors='none',edgecolors='r',linewidths=2.0)
    plt.show()


# 3D test
pos3 = np.load("../full_device_test/3d/init/pos.npy")
nn3 = np.load("../full_device_test/3d/init/nn3.npy")
Nx = Ny = Nz = 20
N = Nx*Ny*Nz
edge_size = Ny*Nz
sample_size = 100
rcut = 5

sample_inds = rng.integers(N,size=sample_size)

check = np.zeros(sample_size,dtype=bool)

for i,k in enumerate(sample_inds):
    print("pos[k,:] ", pos3[k,:])
    neighbs = pos3[nn3[k,:],:]
    print("Neighbours = ")
    print(neighbs)
    dd = np.linalg.norm(neighbs - pos3[k,:], axis=1) 
    check = np.all(dd <= rcut)
    print(check)
