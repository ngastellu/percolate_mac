#!/usr/bin/env python

from os import path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from qcnico import plt_utils
from matplotlib import rcParams

plt_utils.setup_tex()
rcParams['font.size'] = 22

datadir = path.expanduser("~/Desktop/simulation_outputs/percolation/40x40/monte_carlo/percolation_times_dipole/")

datfiles = glob(datadir + "dip*.npy")
nfiles = len(datfiles)
print(nfiles)

temps = np.arange(70,510,5,dtype=np.float64) #temperatures for which MC calcs were conducted

tpercs = np.zeros((nfiles,temps.shape[0]),dtype=float)

for k, datf in enumerate(datfiles):
    tpercs[k,:] = np.load(datf)

tavg = np.mean(tpercs,axis=0)
tstd = np.std(tpercs,axis=0)

plt.plot(temps, tavg, 'r-', lw=0.8, label="$\langle t \\rangle$")
plt.plot(temps, tavg+tstd, 'k--', lw=0.8, label="$\langle t \\rangle\pm\sigma_{t}$")
plt.plot(temps, tavg-tstd, 'k--', lw=0.8)
plt.xlabel("Temperature [K]")
plt.ylabel("Percolation time [MC steps]")
plt.legend()
plt.show()

plt.plot(1000.0/temps, np.log(tavg), 'r-', lw=0.8)
plt.xlabel(" $1000/T$ [K]")
plt.ylabel("$\log\langle t\\rangle$ [MC steps]")
plt.show()
