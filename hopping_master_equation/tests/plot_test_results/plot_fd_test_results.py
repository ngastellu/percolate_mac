#!/usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt


nx = ny = nz = 20

coords = np.load("../full_device_test/2d/pos.npy")
e3 = np.load("../full_device_test/2d/energies-10.npy")
p03 = np.load("../full_device_test/2d/Pinits-10.npy")
pf3 = np.load("../full_device_test/2d/Pfinals-10.npy")


plt.scatter(*coords.T, c=e3)
plt.suptitle(f"Energies")
plt.colorbar()
plt.show()

plt.scatter(*coords.T, c=p03)
plt.suptitle(f"P0")
plt.colorbar()
plt.show()

plt.scatter(*coords.T, c=pf3)
plt.suptitle(f"Pfinal")
plt.colorbar()
plt.show()


pos3 = np.load("../full_device_test/3d/pos.npy")
e3 = np.load("../full_device_test/3d/energies-10.npy")
p03 = np.load("../full_device_test/3d/Pinits-10.npy")
pf3 = np.load("../full_device_test/3d/Pfinals-10.npy")

xslices = [0,1,2,10,11,17,18,19]

for ix in xslices:
    coords = pos3[ix*ny*nz:(ix+1)*(ny*nz),1:]
    print(f"******* ix = {ix} *******")
    print(pos3[ix*ny*nz:(ix+1)*(ny*nz),:])
    plt.scatter(*coords.T, c=e3[ix*ny*nz:(ix+1)*(ny*nz)])
    plt.suptitle(f"Energies of x = {pos3[ix*(ny*nz),0]} slice")
    plt.colorbar()
    plt.show()

    plt.scatter(*coords.T, c=p03[ix*ny*nz:(ix+1)*(ny*nz)])
    plt.suptitle(f"P0 of x = {pos3[ix*(ny*nz),0]} slice")
    plt.colorbar()
    plt.show()

    plt.scatter(*coords.T, c=pf3[ix:ix+(ny*nz)])
    plt.suptitle(f"Pfinal of x = {pos3[ix*ny*nz,0]} slice")
    plt.colorbar()
    plt.show()