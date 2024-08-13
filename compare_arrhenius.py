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

temps = np.arange(40,440,10)[14:]
# temps = np.arange(200,435,5)

# t1dir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/t1/percolate_output/zero_field/virt_100x100_gridMOs_var_a/'
# tdot25dir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tdot25/percolate_output/zero_field/virt_100x100_gridMOs/'

# run_type = 'eps_rho_1.05e-3'

# pCNNdir = f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/zero_field/virt_100x100_gridMOs_rmax_18.03/'
# tempdot6_dir = f'/Users/nico/Desktop/simulation_outputs/percolation/tempdot6/percolate_output/zero_field/virt_100x100_gridMOs_rmax_121.2/'
# tempdot5_dir = f'/Users/nico/Desktop/simulation_outputs/percolation/tempdot5/percolate_output/zero_field/virt_100x100_gridMOs_rmax_198.69/'

pCNNdir = f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/zero_field/virt_100x100_gridMOs_rmax_18.03_psipow1/'
tempdot6_dir = f'/Users/nico/Desktop/simulation_outputs/percolation/tempdot6/percolate_output/zero_field/virt_100x100_gridMOs_rmax_121.2_psipow1/'
tempdot5_dir = f'/Users/nico/Desktop/simulation_outputs/percolation/tempdot5/percolate_output/zero_field/virt_100x100_gridMOs_rmax_198.69_psipow1/'

# pCNNdir = f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/zero_field/virt_100x100_gridMOs_eps_rho_1.05e-3/'
# tempdot6_dir = f'/Users/nico/Desktop/simulation_outputs/percolation/tempdot6/percolate_output/zero_field/virt_100x100_gridMOs_eps_rho_1.05e-3/'
# tempdot5_dir = f'/Users/nico/Desktop/simulation_outputs/percolation/tempdot5/percolate_output/zero_field/virt_100x100_gridMOs_eps_rho_1.05e-3/'



# tdot25_lbls = list(set(range(30)) - {0, 5, 8, 18, 20, 21, 25, 26})

with open(pCNNdir + f'good_runs_rmax_18.03_psipow1.txt') as fo:
    pCNN_lbls = [int(l.strip()) for l in fo.readlines()]

# pCNN_lbls = np.array(pCNN_lbls)
# pCNN_lbls = pCNN_lbls[pCNN_lbls <= 150]
# with open(t1dir + 'good_runs.txt') as fo:
    # t1_lbls = [int(l.strip()) for l in fo.readlines()]
with open(tempdot6_dir + f'good_runs_rmax_121.2_psipow1.txt') as fo:
    tdot6_lbls = [int(l.strip()) for l in fo.readlines()]

with open(tempdot5_dir + f'good_runs_rmax_198.69_psipow1.txt') as fo:
    tdot5_lbls = [int(l.strip()) for l in fo.readlines()]

# tdot5_lbls = range(31)

# tdot6_lbls = list(set(range(2,132)) - {8, 11, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32})

# tdot5_lbls = range(117)


# ddirs = [tdot25dir, pCNNdir, t1dir, tempdot6_dir]
# run_lbls = [tdot25_lbls,pCNN_lbls,t1_lbls,tdot6_lbls]
# run_lbls = [pCNN_lbls,tdot5_lbls]
# curve_lbls = ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']

ddirs = [pCNNdir,tempdot6_dir,tempdot5_dir]
curve_lbls = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
run_lbls = [pCNN_lbls,tdot6_lbls,tdot5_lbls]
clrs = MAC_ensemble_colours()

# ddirs = [tempdot6_dir,tempdot5_dir]
# curve_lbls = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
# run_lbls = [tdot6_lbls,tdot5_lbls]
# clrs = MAC_ensemble_colours('two_ensembles')


# curve_lbls = ['$\delta$-aG','$\chi$-aG']
# clrs2 = MAC_ensemble_colours('two_ensembles')
# clrs[0] = clrs2[0]
# clrs[2] = clrs2[1]


ndatasets = len(ddirs)

setup_tex()


rcParams['font.size'] = 20
# rcParams['figure.figsize'] = [12.8,9.6]

fig, ax = plt.subplots()

eas = np.zeros(ndatasets)
errs_lr = np.zeros(ndatasets)

eas2 = np.zeros(ndatasets)
errs_cf = np.zeros(ndatasets)

sigs = []

r_maxs = ['18.03', '121.2', '198.69']

# for k, dd, ll, cc, cl in zip(range(2), ddirs, run_lbls, clrs, curve_lbls):
for k, dd, ll, rr, cc, cl in zip(range(len(ddirs)), ddirs, run_lbls, r_maxs, clrs, curve_lbls):

    # if k == 1: continue
    print(f'~~~~~~~~ Color = {cc} ~~~~~~~~~')
    dcrits = get_dcrits(ll, temps, dd, pkl_prefix=f'out_percolate_rmax_{rr}_psipow1')
    # dcrits = get_dcrits(ll, temps, dd, pkl_prefix=f'out_percolate_{run_type}')
    # sigmas = saddle_pt_sigma(dcrits)
    sigmas, sigmas_err = sigma_errorbar(dcrits)

 
    slope, intercept, x, y, err_lr, interr_lr  = arrhenius_fit(temps, sigmas, inv_T_scale=1000.0, return_for_plotting=True, return_err=True, w0=conv_factor)
    sigs.append(y)
    print(sigmas_err.shape)
    eas[k] = -slope * kB * 1e6 # in meV
    errs_lr[k] = err_lr * kB * 1e6 #error in Ea estimate from `arrhenius_fit`, in meV

    print(f'\n*** {cl} ***')
    print(f'linregress ---> {eas[k]} ± {errs_lr[k]} meV; sigma_0 = {np.exp(intercept)} [S/m] ; slope err = {interr_lr}')
    # print(np.sort(dcrits)[[0,-1]])
    # print(np.max(sigmas_err))

    ax.plot(x,np.exp(y),'o',label=cl,ms=10.0, c=cc)
    # ax.errorbar(1000/temps,sigmas,yerr=sigmas_err,fmt='-o',c=cc,label=cl,ms=5.0)
    ax.plot(x, np.exp(x*slope+intercept),'-',c=cc,lw=1.6)

  
ax.set_yscale('log')
ax.set_xlabel('$1000/T$ [K$^{-1}$]')
ax.set_ylabel('$\sigma$ [S/m]')
# ax.set_title('Percolation with site size cutoff determined by largest crystallite area')

plt.legend()
plt.show()


# print(p6c)

# ii = np.argsort(p6c)

# fig, ax = plt.subplots()
# # ax.plot(p6c, eas,'ro',ms=5.0)
# # ax.plot(p6c,eas, 'r-',lw=0.8)
# ax.errorbar(p6c[ii],eas[ii],yerr=errs_lr[ii],fmt='-o',label='linregress')
# # ax.errorbar(p6c,eas2,yerr=errs_cf,fmt='-o',label='curve fit')

# ax.set_xlabel('Proprtion of crystalline hexagons $p_{6c}$')
# ax.set_ylabel('$E_{a}$ [meV]')
# # plt.legend()
# plt.show()

# plt.plot(x,sigs[1]/sigs[0],'-o')
# plt.show()
