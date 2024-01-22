#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import linregress
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from qcnico.coords_io import read_xsf
from qcnico.plt_utils import setup_tex, get_cm
from glob import glob




tpercdir = "/Users/nico/Desktop/simulation_outputs/percolation/40x40/monte_carlo/miller_abrahams/percolation_times/"
temps = np.arange(70,505,5,dtype=np.float64)
T_inv = 1000/temps
plot_T_inds = ((T_inv >= 2.5) * (T_inv <= 5.5)).nonzero()[0]
omega0 = 1e11
tfiles = glob(tpercdir + '*npy')

times = np.load(tfiles[0])
for tf in tfiles[1:]:
    times += np.load(tf)

times /= len(tfiles)

kB = 8.617e-5
prop_constant = kB * temps / omega0

avg_times = np.cumsum(times,axis=0) / np.arange(1,1001)[:,None] * prop_constant
diffs = np.abs(np.diff(avg_times,axis=0))

clrs = get_cm(np.arange(1,87),cmap_str='inferno')

setup_tex(fontsize=20)
rcParams['figure.figsize'] = [5.8,4.8]

sample_T_inds = plot_T_inds[[0,-1]]

fig, ax = plt.subplots()

nstart = 200

for i in sample_T_inds:
    ax.plot(np.arange(nstart+1,1000),diffs[nstart:,i],'-',lw=0.8,c=clrs[i],label=f'{int(temps[i])} K')

ax.set_xlabel('\# of MC runs $N_r$')
ax.set_ylabel('$|\langle\\tau\\rangle_{N_r} - \langle\\tau\\rangle_{1000}|$')
plt.legend()
plt.show()

# Arrhenius plot
x = 1000/temps
y = np.log(1.0/avg_times[-1,:])
x = x[plot_T_inds]
y = y[plot_T_inds]

slope, intercept, r, *_ = linregress(x,y)
plt.figure()
plt.plot(x,y, 'ko')
plt.plot(x,y, 'k-', lw=0.8)
plt.plot(x, x*slope + intercept,'r-')
plt.xlabel("$1000/T$ [K]")
plt.ylabel("$\log\sigma$ [$\log\\text{S}$]")
plt.suptitle('Activated hopping $\\rho\sim e^{E_a/k_BT}$')
plt.show()

kB = 8.617e-5
print(f"Ea = {-slope*kB*1000} eV")


# print(slope)


# #Power law plot
# x = np.log(temps)
# y = np.log(rho) + 18
# slope, intercept, r, *_ = linregress(x[x>5.5],y[x>5.5])
# plt.figure()
# plt.plot(x,y, 'ko')
# plt.plot(x,y, 'k-', lw=0.8)
# plt.plot(x, x*slope + intercept,'r-')
# plt.xlabel("$\log T$ [K]")
# plt.suptitle('Rare chain hopping $\\rho\sim T^\\alpha$')
# plt.ylabel("$\log\\rho$")
# plt.show()

# #Mott plot
# x = temps**(-1/3)
# y = np.log(rho) + 18
# slope, intercept, r, *_ = linregress(x[x<0.20],y[x<0.20])
# plt.figure()
# plt.plot(x,y, 'ko')
# plt.plot(x,y, 'k-', lw=0.8)
# plt.plot(x, x*slope + intercept,'r-')
# plt.suptitle('Mott fit $\\rho\sim e^{-(T_0/T)^{1/3}}$')
# plt.xlabel("$T^{-1/3}$ [K]")
# plt.ylabel("$\log\\rho$")
# plt.show()
