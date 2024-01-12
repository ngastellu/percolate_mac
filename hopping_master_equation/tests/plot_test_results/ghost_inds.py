#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import get_cm


def check_interval(i,edge_size,xslice):
    return i >= edge_size * xslice and i < edge_size * (xslice+1)

ighost = np.load('../full_device_test/3d/init/ighost.npy') - 1
pos = np.load('../full_device_test/3d/init/pos.npy')
energies = np.load('../full_device_test/3d/init/energies.npy')
pinit = np.load('../full_device_test/3d/pinits-10.npy')
pfinal = np.load('../full_device_test/3d/pfinals-10.npy')

Nx = Ny = Nz = 20
a = 10
edge_size = Ny*Nz
L = np.array([Ny-1,Nz-1]) * a


x_slice = 12
pos_slice  = pos[x_slice*edge_size:(x_slice+1)*edge_size,1:]
print(pos_slice)

plt.scatter(*pos_slice.T,c='k')

clrs = get_cm(np.arange(40),'viridis')
k=0

for ij in enumerate(ighost):
    i,j = ij
    if check_interval(i,edge_size,x_slice) and check_interval(j,edge_size,x_slice):
        print(f"Ghost site {i} --> {pos[i,:]}")
        print(f"Real site {j} --> {pos[j,:]}")
        print(f"Energies match = {energies[i] == energies[j]}\n")
        print(f"P0s match = {pinit[i] == pinit[j]}\n")
        print(f"Pfinals match = {pfinal[i] == pfinal[j]}\n")
        plt.scatter(pos[i,1],pos[i,2], c=clrs[k])
        plt.scatter(pos[j,1],pos[j,2],color=clrs[k])
        k += 1

plt.show()

print("\n\nNow looping overall ALL ghost sites array!\n")

for ij in ighost:
    i,j = ij 
    ematch = energies[i] == energies[j]
    pimatch = pinit[i] == pinit[j]
    pfmatch = pfinal[i] == pfinal[j]
    match_bools = np.array((ematch,pimatch,pfmatch))
    if np.any(~match_bools):
        print(f"Ghost site {i} --> {pos[i,:]}")
        print(f"Real site {j} --> {pos[j,:]}")
        print(f"Energies match = {ematch}")
        print(f"P0s match = {pimatch}")
        print(f"Pfinals match = {pfmatch}")
    
rtest1 = np.array([80,90,190])
rtest2 = np.array([80,50,190])
rtest3 = np.array([80,190,130])

rtests = [rtest1,rtest2, rtest3]


for rt in rtests:
    print(f"Working on {rt}")
    match = np.all(pos == rt, axis=1).nonzero()[0][0]
    ghost_check = (ighost[:,0] == match).nonzero()[0]
    for g in ghost_check:
        k = ighost[g,1]
        print(pos[k,:])
    print('\n')