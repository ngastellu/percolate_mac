#!/usr/bin/env python

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import linregress
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from qcnico.coords_io import read_xsf
from qcnico.plt_utils import setup_tex




tpercdir = "/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/marcus/percolation_times_dipole/MC_1000/"
strucdir = "/Users/nico/Desktop/simulation_outputs/percolation/40x40/structures/"
struc = remove_dangling_carbons(read_xsf(strucdir+'bigMAC-150_relaxed.xsf')[0],1.8)
omega0 = 1e11
dX = np.max(struc[:,0]) - np.min(struc[:,0])
E = 1.0/dX
temps = np.arange(70,505,5,dtype=np.float64)
T_inv = 1000/temps
plot_T_inds = ((T_inv >= 2.5) * (T_inv <= 5.5)).nonzero()[0]
dat = np.array([np.load(npy) for npy in glob(tpercdir + 'dipole_perc_times-*.npy')])
times = np.mean(dat,axis=0)
print(times.shape)
sigma = np.std(dat,axis=0)

# Regular plot
# plt.figure()
# plt.plot(temps, times, 'r-', lw=0.8,label="$\langle \\tau\\rangle$")
# # plt.plot(temps, times+sigma, 'k--', lw=0.8,label="$\langle \\tau\\rangle\pm\sigma_\\tau$")
# # plt.plot(temps, times-sigma, 'k--', lw=0.8)
# plt.xlabel("$T$ [K]")
# plt.ylabel(" Hopping time [fs]")
# plt.legend()
# plt.show()


kB = 8.617e-5
rho = (E*np.mean(times,axis=0))/dX
prop_constant = kB * temps / omega0
rho = rho * prop_constant

print(rho.shape)

setup_tex(fontsize=20)
rcParams['figure.figsize'] = [5.8,4.8]

# Arrhenius plot
x = 1000/temps
y = np.log(1.0/rho)
x = x[plot_T_inds]
y = y[plot_T_inds]
print(y.shape)
slope, intercept, r, *_ = linregress(x,y)
plt.figure()
plt.plot(x,y, 'ko')
plt.plot(x,y, 'k-', lw=0.8)
plt.plot(x, x*slope + intercept,'r-')
plt.xlabel("$1000/T$ [K]")
plt.ylabel("$\log\sigma$ [$\log\\text{S}$]")
plt.suptitle('Activated hopping $\\rho\sim e^{E_a/k_BT}$')
plt.show()


print(f"Ea = {-slope*kB*1000} eV")


#Power law plot
x = np.log(temps)
y = np.log(rho) + 18
slope, intercept, r, *_ = linregress(x[x>5.5],y[x>5.5])
plt.figure()
plt.plot(x,y, 'ko')
plt.plot(x,y, 'k-', lw=0.8)
plt.plot(x, x*slope + intercept,'r-')
plt.xlabel("$\log T$ [K]")
plt.suptitle('Rare chain hopping $\\rho\sim T^\\alpha$')
plt.ylabel("$\log\\rho$")
plt.show()

#Mott plot
x = temps**(-1/3)
y = np.log(rho) + 18
slope, intercept, r, *_ = linregress(x[x<0.20],y[x<0.20])
plt.figure()
plt.plot(x,y, 'ko')
plt.plot(x,y, 'k-', lw=0.8)
plt.plot(x, x*slope + intercept,'r-')
plt.suptitle('Mott fit $\\rho\sim e^{-(T_0/T)^{1/3}}$')
plt.xlabel("$T^{-1/3}$ [K]")
plt.ylabel("$\log\\rho$")
plt.show()
