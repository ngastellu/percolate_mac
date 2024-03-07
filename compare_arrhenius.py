#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qcnico.plt_utils import setup_tex, get_cm
from utils_analperc import get_dcrits, saddle_pt_sigma, arrhenius_fit

def sigma_errorbar(dcrits):
    """IDEA: Estimate uncertainty in sigma(T) for each T by omitting one structure from the data set
     before computing sigma, and taking the difference between sigma obtained from this reduced set
     and sigma the computed from the full data set. We cycle over all samples and keep the highest 
     difference between the two estimates of sigma as our uncertainty.
     As always, dcrits is (Ns x Nt) array where Ns = number of structures, and
     Nt = number of temperatures."""
    
    nsamples = dcrits.shape[0]
    sigmas_full = saddle_pt_sigma(dcrits)
    err = np.zeros(temps.shape[0])

    # Get errors in sigma estimates
    for n in range(nsamples):
        sigmas_partial = saddle_pt_sigma(np.roll(dcrits, n, axis=0)[:-1])
        diffs = np.abs(sigmas_full - sigmas_partial)
        inew = (diffs > err).nonzero()[0]
        err[inew] = diffs[inew]
    
    return sigmas_full, err

kB = 8.617e-5
# temps = np.arange(40,440,10)[14:]
temps = np.arange(200,435,5)

tdot25dir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tdot25/percolate_output/zero_field/virt_100x100_gridMOs/finer_T_grid/'
pCNNdir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/zero_field/100x100_gridMOs/finer_T_grid/'
t1dir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/t1/percolate_output/zero_field/virt_100x100_gridMOs/finer_T_grid/'



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
fig, ax = plt.subplots()

eas = np.zeros(3)
errs_lr = np.zeros(3)

eas2 = np.zeros(3)
errs_cf = np.zeros(3)

for k, dd, ll, cc, cl in zip(range(3), ddirs, run_lbls, clrs, curve_lbls):
    # if k == 1: continue

    dcrits = get_dcrits(ll, temps, dd)

    # sigmas = saddle_pt_sigma(dcrits)
    sigmas, sigmas_err = sigma_errorbar(dcrits)

 
    slope, intercept, x, y, err_lr, interr_lr  = arrhenius_fit(temps, sigmas, inv_T_scale=1000.0, return_for_plotting=True, return_err=True)
    eas[k] = -slope * kB * 1e6 # in meV
    errs_lr[k] = err_lr * kB * 1e6 #error in Ea estimate from `arrhenius_fit`, in meV

    print(f'\n*** {cl} ***')
    print(f'linregress ---> {eas[k]} ± {errs_lr[k]} meV; slope err = {interr_lr}')
    print(np.max(sigmas_err))

    # ax.plot(x,y,'o',c=cc,label=cl,ms=5.0)
    ax.errorbar(1000/temps,sigmas,yerr=sigmas_err,fmt='-o',c=cc,label=cl,ms=5.0)
    # ax.plot(x, x*slope+intercept,'-',c=cc,lw=0.8)

  
ax.set_xlabel('$1000/T$ [K$^{-1}$]')
ax.set_ylabel('$\sigma$')
ax.set_title('Virtual near-$E_F$ MOs')

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
