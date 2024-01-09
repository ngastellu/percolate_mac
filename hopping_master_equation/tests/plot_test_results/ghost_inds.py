#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def check_interval(i,edge_size,xslice):
    return i >= edge_size * xslice and i < edge_size * (xslice+1)

ighost = np.load('../full_device_test/3d/init/ighost.npy')
pos = np.load('../full_device_test/3d/init/pos.npy')
Nx = Ny = Nz = 20
edge_size = Ny*Nz


x_slice = 12
pos_slice  = pos[x_slice*edge_size:(x_slice+1)*edge_size,1:]
print(pos_slice)

plt.scatter(*pos_slice.T,c='k')

for ij in ighost:
    i,j = ij - 1
    if check_interval(i,edge_size,x_slice) and check_interval(j,edge_size,x_slice):
        print(f"Ghost site {i} --> {pos[i,:]}")
        print(f"Real site {j} --> {pos[j,:]}\n")
        plt.scatter(pos[i,1],pos[i,2], c='red')
        plt.scatter(pos[j,1],pos[j,2],color='green')

plt.show()

