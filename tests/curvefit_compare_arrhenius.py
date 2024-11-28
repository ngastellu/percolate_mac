#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qcnico.plt_utils import setup_tex, get_cm
from utils_analperc import get_dcrits, saddle_pt_sigma, arrhenius_fit


def arrhenius(T,Ea,A):
    kB = 8.617333262e-5 #eV/K
    return A * np.exp(-Ea/(kB*T))


def arrhenius_fit_cf(T,sigmas):
    popt, pcov = curve_fit(arrhenius,T,sigmas)
    return popt, pcov




kB = 8.617e-5
temps = np.arange(40,440,10)[14:]

tdot25dir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tdot25/percolate_output/zero_field/virt_100x100_gridMOs/'
pCNNdir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/zero_field/100x100_gridMOs/'
t1dir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/t1/percolate_output/zero_field/virt_100x100_gridMOs/'


tdot25_lbls = list(set(range(30)) - {0, 5, 8, 18, 20, 21, 25, 26})

with open(pCNNdir + 'good_runs.txt') as fo:
    pCNN_lbls = [int(l.strip()) for l in fo.readlines()]

with open(t1dir + 'good_runs.txt') as fo:
    t1_lbls = [int(l.strip()) for l in fo.readlines()]

dd_rings = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'

ring_data_tdot25 = np.load(dd_rings + 'avg_ring_counts_tdot25_relaxed.npy')
ring_data_t1 = np.load(dd_rings + 'avg_ring_counts_t1_relaxed.npy')
ring_data_pCNN = np.load(dd_rings + 'avg_ring_counts_normalised.npy')

p6c_tdot25 = ring_data_tdot25[3] / ring_data_tdot25.sum()
p6c_t1 = ring_data_t1[3] / ring_data_t1.sum()
p6c_pCNN = ring_data_pCNN[3]
p6c = np.array([p6c_tdot25, p6c_pCNN,p6c_t1])

clrs = get_cm(p6c, 'inferno')
ddirs = [tdot25dir, pCNNdir, t1dir]
run_lbls = [tdot25_lbls,pCNN_lbls,t1_lbls]
curve_lbls = ['$\\tilde{T} = 0.25$', 'PixelCNN', '$\\tilde{T} = 1$']

setup_tex()
fig, ax = plt.subplots(2,1)

eas = np.zeros(3)
errs_lr = np.zeros(3)

eas2 = np.zeros(3)
errs_cf = np.zeros(3)

for k, dd, ll, cc, cl in zip(range(3), ddirs, run_lbls, clrs, curve_lbls):
    # if k == 1: continue

    dcrits = get_dcrits(ll, temps, dd)

    sigmas = saddle_pt_sigma(dcrits)

    popt, stderr = arrhenius_fit_cf(temps, sigmas)
    eas2[k] = popt[0] * 1e3
    errs_cf[k] = stderr[0,0] * 1e3
    interr_cf = stderr[1,1]
    
    slope, intercept, x, y, err_lr, interr_lr  = arrhenius_fit(temps, sigmas, inv_T_scale=1000.0, return_for_plotting=True, return_err=True)
    eas[k] = -slope * kB * 1e6 # in meV
    errs_lr[k] = err_lr * kB * 1e6 #error in Ea estimate from `arrhenius_fit`, in meV

    print(f'\n*** {cl} ***')
    print(f'linregress ---> {eas[k]} ± {errs_lr[k]} meV; slope err = {interr_lr}')
    print(f'curve_fit ---> {eas2[k]} ± {errs_cf[k]} meV; slope err = {interr_cf}')

    ax[0].plot(x,y,'o',c=cc,label=cl,ms=5.0)
    ax[0].plot(x, x*slope+intercept,'-',c=cc,lw=0.8)

    ax[1].plot(1000.0/temps,y,'o',c=cc,label=cl,ms=5.0)
    ax[1].plot(1000/temps,np.log(arrhenius(temps,popt[0],popt[1])),'-',c=cc,lw=0.8)
  
ax[1].set_xlabel('$1000/T$ [K$^{-1}$]')
ax[0].set_ylabel('$\log\sigma$')
ax[1].set_ylabel('$\log\sigma$')
ax[0].set_title('Virtual near-$E_F$ MOs')

plt.legend()
plt.show()

fig, ax = plt.subplots()
# ax.plot(p6c, eas,'ro',ms=5.0)
# ax.plot(p6c,eas, 'r-',lw=0.8)
ax.errorbar(p6c,eas,yerr=errs_lr,fmt='-o',label='linregress')
# ax.errorbar(p6c,eas2,yerr=errs_cf,fmt='-o',label='curve fit')

ax.set_xlabel('$p_{6c}$')
ax.set_ylabel('$E_{a}$ [meV]')
# plt.legend()
plt.show()
