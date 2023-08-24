#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from glob import glob
from qcnico import plt_utils



datadir = "/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/percolation_times_dipole/MC_1000/"
dipfiles = glob(datadir + "dip*npy")

temps = np.arange(70,505,5,dtype=np.float64)

times = np.zeros_like(temps)

# for f in dipfiles:
#     dat = np.load(f)
#     times += np.mean(dat,axis=0)


# times /= len(dipfiles)

# plt_utils.setup_tex()
# rcParams['font.size'] = 24


# # Regular plot
# plt.plot(temps, times, 'r-', lw=0.8)
# plt.xlabel("$T$ [K]")
# plt.ylabel("$\langle t_{perc}\\rangle$ [fs]")
# plt.show()

# # Arrhenius plot
# plt.plot(1000/temps, np.log(times), 'r-', lw=0.8)
# plt.xlabel("$1000/T$ [K]")
# plt.ylabel("$\log\langle t_{perc}\\rangle$ [fs]")
# plt.show()



# Plot individual realisations
np.random.seed(42)
inds = np.random.randint(len(dipfiles), size=10)

for i in inds:

    dat = np.load(dipfiles[i])
    times = np.mean(dat,axis=0)
    print(times.shape)
    sigma = np.std(dat,axis=0)

    # Regular plot
    plt.plot(temps, times, 'r-', lw=0.8,label="$\langle t\\rangle$")
    plt.plot(temps, times+sigma, 'k--', lw=0.8,label="$\langle t\\rangle\pm\sigma_t$")
    plt.plot(temps, times-sigma, 'k--', lw=0.8)
    plt.xlabel("$T$ [K]")
    plt.ylabel(" Time [fs]")
    plt.legend()
    plt.show()

    # Arrhenius plot
    plt.plot(1000/temps, np.log(times), 'r-', lw=0.8)
    plt.xlabel("$1000/T$ [K]")
    plt.ylabel("$\log\langle t_{perc}\\rangle$ [fs]")
    plt.show()



