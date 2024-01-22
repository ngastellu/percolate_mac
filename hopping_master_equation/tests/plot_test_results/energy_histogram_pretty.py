#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico import plt_utils



Nx = Ny = 20
N = Nx*Ny
energies = np.load("../full_device_test/2d/init/energies.npy")[2*Ny:N-2*Ny]

clrs = plt_utils.get_cm(energies,cmap_str='viridis',max_val=1.0)

plt_utils.setup_tex(fontsize=30)

fig, ax = plt.subplots()

hist, bins = np.histogram(energies, 50)
hist = hist / energies.size
centers = (bins[1:] + bins[:-1])/2
clrs = plt_utils.get_cm(centers,cmap_str='viridis',max_val=1.0)
dx = bins[1:] - bins[:-1]

x = np.linspace(np.min(energies),np.max(energies),1000)
nu = 0.007
y = np.exp(-(x**2)/(2*nu)) / (nu*np.sqrt(2*np.pi))

ax.bar(centers, hist, align='center', width=dx, color=clrs)
# ax.plot(x,y,'k-',lw=0.8)
ax.set_xlabel("$\\varepsilon_i$")
ax.set_ylabel("Counts")
plt.show()

