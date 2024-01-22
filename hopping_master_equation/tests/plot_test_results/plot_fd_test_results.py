#!/usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex


nx = ny = nz = 20

setup_tex(fontsize=30)

coords = np.load("../full_device_test/2d/init/pos.npy")
e3 = np.load("../full_device_test/2d/init/energies.npy")
# p03 = np.load("../full_device_test/2d/Pinits-10.npy")
# pf3 = np.load("../full_device_test/2d/Pfinals-10.npy")

plt.scatter(*coords.T, c=e3, s=80.0)
plt.scatter(*coords[:2*ny].T, c='gray',s=120.0)
plt.scatter(*coords[-2*ny:].T, c='gray',s=120.0)
plt.xlabel("$x$ [\AA]")
plt.ylabel("$y$ [\AA]")
plt.suptitle(f"Energies")
plt.colorbar()
plt.show()

# plt.scatter(*coords.T, c=p03)
# plt.suptitle(f"P0")
# plt.colorbar()
# plt.show()

# plt.scatter(*coords.T, c=pf3)
# plt.suptitle(f"Pfinal")
# plt.colorbar()
# plt.show()


# pos3 = np.load("../full_device_test/3d/init/pos.npy")
# e3 = np.load("../full_device_test/3d/init/energies.npy")
# p03 = np.load("../full_device_test/3d/init/prob.npy")

# pos3 = np.load("../full_device_test/3d/pos.npy")
# e3 = np.load("../full_device_test/3d/energies-10.npy")
# pi3 = np.load("../full_device_test/3d/Pinits-10.npy")
# p03 = np.load("../full_device_test/3d/p0.npy")
# pf3 = np.load("../full_device_test/3d/Pfinals-10.npy")
# # xinds = range(nx)
# xinds = [0,1,2,5,9,10,17,18,19]

# for ix in xinds:
#     xslice = slice(ix*ny*nz,(ix+1)*ny*nz)
#     coords = pos3[xslice,1:]
#     print(f"******* ix = {ix} *******")
#     print(pos3[xslice,:])
#     print(np.all(pos3[xslice,0] == pos3[ix*ny*nz,0]))
#     plt.scatter(*coords.T, c=e3[xslice])
#     plt.suptitle(f"Energies of x = {pos3[ix*(ny*nz),0]} slice")
#     plt.colorbar()
#     plt.show()

#     plt.scatter(*coords.T, c=p03[xslice])
#     plt.suptitle(f"P0 of x = {pos3[ix*(ny*nz),0]} slice")
#     plt.colorbar()
#     plt.show()

#     # plt.scatter(*coords.T, c=pf3[xslice])
#     # plt.suptitle(f"Pfinal of x = {pos3[ix*ny*nz,0]} slice")
#     # plt.colorbar()
#     # plt.show() 