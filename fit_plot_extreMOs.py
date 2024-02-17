#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex
from utils_analperc import get_dcrits, saddle_pt_sigma, arrhenius_fit, mott_fit, find_best_fit


datadir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/extremal_MOs/hi50/'
temps = np.arange(40,440,10)

kB = 8.617333262e-5 #eV/K

with open(datadir + 'good_runs.txt') as fo:
    run_lbls = [int(l.strip()) for l in fo.readlines()]

dcrits = get_dcrits(run_lbls,temps,datadir)
print(dcrits)
sigmas = saddle_pt_sigma(dcrits)


print((sigmas == 0).nonzero())

temps = temps[sigmas > 0]
print(f'Minimum T yielding finite sigma = {min(temps)} K')
sigmas = sigmas[sigmas > 0]


slope, intercept, x, y = arrhenius_fit(temps,sigmas,inv_T_scale=1000.0,return_for_plotting=True)
Ea = -slope*kB*1e6


setup_tex()

fig, ax = plt.subplots()
ax.plot(x,y,'ko',ms=5.0)
ax.plot(x,slope*x+intercept,'r-',lw=0.8)
ax.set_title(f'Arrhenius fit: $E_a = {Ea}$ meV')
ax.set_xlabel("$1000/T$ [K$^{-1}$]")
ax.set_ylabel("$\log \sigma$ [$\log\\text{S}$]")
plt.show()


slope, intercept, x, y = mott_fit(temps,sigmas, return_for_plotting=True, x_end=8)
T0 = -slope**3

setup_tex()

fig, ax = plt.subplots()
ax.plot(x,y,'ko',ms=5.0)
ax.plot(x,slope*x+intercept,'r-',lw=0.8)
ax.set_title(f'Mott 2D fit: $T_0 = {T0}$ K')
ax.set_xlabel("$T^{-1/3}$ [K$^{-1}$]")
ax.set_ylabel("$\log \sigma$ [$\log\\text{S}$]")
plt.show()


slope, intercept, x, y = mott_fit(temps,sigmas,d=1,return_for_plotting=True)
T0 = -slope**2

setup_tex()

fig, ax = plt.subplots()
ax.plot(x,y,'ko',ms=5.0)
ax.plot(x,slope*x+intercept,'r-',lw=0.8)
ax.set_title(f'Mott 1D fit: $T_0 = {T0}$ K')
ax.set_xlabel("$T^{-1/2}$ [K$^{-1/2}$]")
ax.set_ylabel("$\log \sigma$ [$\log\\text{S}$]")
plt.show()

