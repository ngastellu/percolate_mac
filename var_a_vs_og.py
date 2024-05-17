#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
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
        inew = (diffs > err).nonzero()[0] #identify for which T inds we need to keep err 
        err[inew] = diffs[inew]
    
    return sigmas_full, err



kB = 8.617e-5
e2C = 1.602177e-19 # elementary charge to Coulomb
w0 = 1e10
# This factor combines the hop frequency with the unit conversion (to yield conductivity in siemens)
# w0 is chosen such that the final result matches the results from the AMC paper.
conv_factor = e2C*w0

temps = np.arange(40,440,10)[14:]
# temps = np.arange(200,435,5)

tdot6dir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tempdot6/percolate_output/zero_field/'


var_a_dir = tdot6dir +  'virt_100x100_gridMOs_rollback/'
og_dir =  tdot6dir +  'virt_100x100_gridMOs/'

with open(og_dir + 'good_runs.txt') as fo:
    run_lbls = [int(l.strip()) for l in fo.readlines()]


#run_lbls = range(31)

# run_lbls = range(132)

cyc = rcParams['axes.prop_cycle'] #default plot colours are stored in this `cycler` type object
clrs = [d['color'] for d in list(cyc[0:3])]

# ddirs = [tdot25dir, pCNNdir, t1dir, tempdot6_dir]
ddirs = [var_a_dir, og_dir]
curve_lbls = ['var a', 'OG']

ndatasets = len(ddirs)

setup_tex()
fig, ax = plt.subplots()

sigs = []

ll = run_lbls
# for k, dd, ll, cc, cl in zip(range(2), ddirs, run_lbls, clrs, curve_lbls):
for k, dd, cc, cl in zip(range(len(ddirs)), ddirs, clrs, curve_lbls):
    # if k == 1: continue
    print(f'~~~~~~~~ Color = {cc} ~~~~~~~~~')
    # if k == 0:
    #     dcrits = get_dcrits(ll, temps, dd, var_a=True)
    # else:
    print(k)
    if k == 0:
        dcrits = get_dcrits(ll, temps, dd, rollback=True)
    
    else:
        dcrits = get_dcrits(ll, temps, dd, rollback=False)


    # sigmas = saddle_pt_sigma(dcrits)
    sigmas, sigmas_err = sigma_errorbar(dcrits)

    sigs.append(sigmas)
    slope, intercept, x, y, err_lr, interr_lr  = arrhenius_fit(temps, sigmas, inv_T_scale=1000.0, return_for_plotting=True, return_err=True, w0=conv_factor)
    print(sigmas_err.shape)
    ea = -slope * kB * 1e6 # in meV
    err_lr = err_lr * kB * 1e6 #error in Ea estimate from `arrhenius_fit`, in meV

    print(f'\n*** {cl} ***')
    print(f'linregress ---> {ea} ± {err_lr} meV; sigma_0 = {np.exp(intercept)} [S/m] ; slope err = {interr_lr}')
    # print(np.sort(dcrits)[[0,-1]])
    # print(np.max(sigmas_err))

    ax.plot(x,np.exp(y),'o',label=cl,ms=5.0, c=cc)
    # ax.errorbar(1000/temps,sigmas,yerr=sigmas_err,fmt='-o',c=cc,label=cl,ms=5.0)
    ax.plot(x, np.exp(x*slope+intercept),'-',c=cc,lw=0.8)

  
ax.set_yscale('log')
ax.set_xlabel('$1000/T$ [K$^{-1}$]')
ax.set_ylabel('$\sigma$ [S/m]')

plt.legend()
plt.show()

# print(np.all(sigs[0]==sigs[1]))