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

tdot25dir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tdot25/percolate_output/zero_field/virt_100x100_gridMOs/'
pCNNdir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/zero_field/virt_100x100_gridMOs_var_a_rollback/'
t1dir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/t1/percolate_output/zero_field/virt_100x100_gridMOs_var_a/'
tempdot6_dir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tempdot6/percolate_output/zero_field/virt_100x100_gridMOs_var_a_hl/'
tempdot5_dir = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tempdot5/percolate_output/zero_field/virt_100x100_gridMOs_var_a_hl/'



tdot25_lbls = list(set(range(30)) - {0, 5, 8, 18, 20, 21, 25, 26})

with open(pCNNdir + 'good_runs_var_a.txt') as fo:
    pCNN_lbls = [int(l.strip()) for l in fo.readlines()]

pCNN_lbls = np.array(pCNN_lbls)
pCNN_lbls = pCNN_lbls[pCNN_lbls <= 150]
# with open(t1dir + 'good_runs.txt') as fo:
    # t1_lbls = [int(l.strip()) for l in fo.readlines()]
with open(tempdot6_dir + 'good_runs_var_a.txt') as fo:
    tdot6_lbls = [int(l.strip()) for l in fo.readlines()]

with open(tempdot5_dir + 'good_runs_var_a_hl.txt') as fo:
    tdot5_lbls = [int(l.strip()) for l in fo.readlines()]

# tdot5_lbls = range(31)

# tdot6_lbls = list(set(range(2,132)) - {8, 11, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32})

# tdot5_lbls = range(117)

dd_rings = '/Users/nico/Desktop/simulation_outputs/ring_stats_40x40_pCNN_MAC/'

ring_data_tdot25 = np.load(dd_rings + 'avg_ring_counts_tdot25_relaxed.npy')
ring_data_t1 = np.load(dd_rings + 'avg_ring_counts_t1_relaxed.npy')
ring_data_pCNN = np.load(dd_rings + 'avg_ring_counts_normalised.npy')
ring_data_tempdot6 = np.load(dd_rings + 'avg_ring_counts_tempdot6_new_model_relaxed.npy')

p6c_tdot25 = ring_data_tdot25[4] / ring_data_tdot25.sum()
p6c_t1 = ring_data_t1[3] / ring_data_t1.sum()
p6c_tempdot6 = ring_data_tempdot6[4] / ring_data_tempdot6.sum()
p6c_pCNN = ring_data_pCNN[3]
# p6c = np.array([p6c_tdot25, p6c_pCNN,p6c_t1,p6c_tempdot6])
p6c = np.array([p6c_pCNN,p6c_tempdot6])

# clrs = get_cm(p6c, 'inferno',min_val=0.25,max_val=0.7)


cyc = rcParams['axes.prop_cycle'] #default plot colours are stored in this `cycler` type object
clrs = [d['color'] for d in list(cyc[0:3])]

# ddirs = [tdot25dir, pCNNdir, t1dir, tempdot6_dir]
ddirs = [pCNNdir, tempdot6_dir, tempdot5_dir]
# run_lbls = [tdot25_lbls,pCNN_lbls,t1_lbls,tdot6_lbls]
run_lbls = [pCNN_lbls,tdot6_lbls,tdot5_lbls]
# curve_lbls = ['$\\tilde{T} = 0.25$', 'PixelCNN', '$\\tilde{T} = 1$', '$\\tilde{T} = 0.6$']
# curve_lbls = ['$\\tilde{T} = 0.25$', 'PixelCNN', '$\\tilde{T} = 0.6$']
# curve_lbls = [f'$p_{{6c}} = {p*100:.2f}\,\%$' for p in p6c]
# curve_lbls = ['sAMC-500', 'sAMC-300', 'tdot5']
curve_lbls = ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']

ndatasets = len(ddirs)

setup_tex()
fig, ax = plt.subplots()

eas = np.zeros(ndatasets)
errs_lr = np.zeros(ndatasets)

eas2 = np.zeros(ndatasets)
errs_cf = np.zeros(ndatasets)

sigs = []

# for k, dd, ll, cc, cl in zip(range(2), ddirs, run_lbls, clrs, curve_lbls):
for k, dd, ll, cc, cl in zip(range(len(ddirs)), ddirs, run_lbls, clrs, curve_lbls):
    # if k == 1: continue
    print(f'~~~~~~~~ Color = {cc} ~~~~~~~~~')
    if k == 0:
        dcrits = get_dcrits(ll, temps, dd,rollback=True)
    else:
        dcrits = get_dcrits(ll, temps, dd,hyperlocal=True)
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

    ax.plot(x,np.exp(y),'o',label=cl,ms=5.0, c=cc)
    # ax.errorbar(1000/temps,sigmas,yerr=sigmas_err,fmt='-o',c=cc,label=cl,ms=5.0)
    ax.plot(x, np.exp(x*slope+intercept),'-',c=cc,lw=0.8)

  
ax.set_yscale('log')
ax.set_xlabel('$1000/T$ [K$^{-1}$]')
ax.set_ylabel('$\sigma$ [S/m]')
#ax.set_title('Virtual near-$E_F$ MOs')

plt.legend()
plt.show()


print(p6c)

ii = np.argsort(p6c)

fig, ax = plt.subplots()
# ax.plot(p6c, eas,'ro',ms=5.0)
# ax.plot(p6c,eas, 'r-',lw=0.8)
ax.errorbar(p6c[ii],eas[ii],yerr=errs_lr[ii],fmt='-o',label='linregress')
# ax.errorbar(p6c,eas2,yerr=errs_cf,fmt='-o',label='curve fit')

ax.set_xlabel('Proprtion of crystalline hexagons $p_{6c}$')
ax.set_ylabel('$E_{a}$ [meV]')
# plt.legend()
plt.show()

plt.plot(x,sigs[1]/sigs[0],'-o')
plt.show()
