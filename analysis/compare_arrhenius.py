#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours
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

temps_start_ind = 0

sigmas_dir = '/Users/nico/Desktop/simulation_outputs/percolation/sigmas_v_T/'

structypes = ['40x40', 'tempdot6', 'tempdot5']
curve_lbls = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
clrs = MAC_ensemble_colours()


ndatasets = len(structypes)

setup_tex()
fig, ax = plt.subplots()

eas = np.zeros(ndatasets)
errs_lr = np.zeros(ndatasets)

eas2 = np.zeros(ndatasets)
errs_cf = np.zeros(ndatasets)

sigs = []

# for k, dd, ll, cc, cl in zip(range(2), ddirs, run_lbls, clrs, curve_lbls):
for k, st, cc, cl in zip(range(3), structypes, clrs, curve_lbls):
    # if k == 1: continue
    print(f'~~~~~~~~ Color = {cc} ~~~~~~~~~')

    # sigmas = saddle_pt_sigma(dcrits)
    temps, sigmas, sigmas_err = np.load(sigmas_dir + f'sigma_v_T_w_err_{st}_oldschool_virtual_w_HOMO.npy').T    
    # temps = temps[temps_start_ind:]
    # sigmas = sigmas[temps_start_ind:]
    # sigmas_err = sigmas_err[temps_start_ind:]

    # sigmas = 1.0/sigmas



 
    slope, intercept, x, y, err_lr, interr_lr  = arrhenius_fit(temps, sigmas, inv_T_scale=1000.0, return_for_plotting=True, return_err=True, w0=conv_factor)
    sigs.append(y)
    print(sigmas_err.shape)
    eas[k] = -slope * kB * 1e6 # in meV
    errs_lr[k] = err_lr * kB * 1e6 #error in Ea estimate from `arrhenius_fit`, in meV

    print(f'\n*** {cl} ***')
    print(f'linregress ---> {eas[k]} ± {errs_lr[k]} meV; sigma_0 = {np.exp(intercept)} [S/m] ; slope err = {interr_lr}')
    # print(np.sort(dcrits)[[0,-1]])
    # print(np.max(sigmas_err))

    ax.plot(x,np.exp(y),'o',label=cl,ms=5.0, c=cc)
    # ax.errorbar(1000/temps,sigmas,yerr=sigmas_err,fmt='-o',c=cc,label=cl,ms=5.0)
    ax.plot(x, np.exp(x*slope+intercept),'-',c=cc,lw=0.8)

  
ax.set_yscale('log')
ax.set_xlabel('$1000/T$ [K$^{-1}$]')
ax.set_ylabel('$\sigma$ [S/m]')
#ax.set_title('Virtual near-$E_F$ MOs')

plt.legend()
plt.show()
