#!/usr/bin/env python

from os import path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours
from utils_analperc import arrhenius_fit




kB = 8.617e-5
e2C = 1.602177e-19 # elementary charge to Coulomb
w0 = 1e10
# This factor combines the hop frequency with the unit conversion (to yield conductivity in siemens)
# w0 is chosen such that the final result matches the results from the AMC paper.
conv_factor = e2C*w0

sigmasdir = '/Users/nico/Desktop/simulation_outputs/percolation/sigmas_v_T/'
motype = 'lo'
# temps = np.arange(40,440,10)[14:]

r_maxs = ['18.03', '121.2', '198.69']
curve_lbls = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
structypes=['40x40', 'tempdot6', 'tempdot5']

clrs = MAC_ensemble_colours()

ndatasets = len(curve_lbls)

setup_tex()


rcParams['font.size'] = 20
# rcParams['figure.figsize'] = [12.8,9.6]

fig, ax = plt.subplots()

eas = np.zeros(ndatasets)
errs_lr = np.zeros(ndatasets)

eas2 = np.zeros(ndatasets)
errs_cf = np.zeros(ndatasets)

sigs = []


# for k, dd, ll, cc, cl in zip(range(2), ddirs, run_lbls, clrs, curve_lbls):
for k, st, rr, cc, cl in zip(range(ndatasets), structypes, r_maxs, clrs, curve_lbls):

    print(f'~~~~~~~~ Color = {cc} ~~~~~~~~~')
    temps, sigmas, sigmas_err = np.load(path.join(sigmasdir, f'sigma_v_T_w_err_{st}_rmax_{rr}_psipow2_sites_gammas_{motype}.npy')).T

    if k == 1:
        temps = temps[14:]
        sigmas = sigmas[14:]
        sigmas_err = sigmas_err[14:]

 
    slope, intercept, x, y, err_lr, interr_lr  = arrhenius_fit(temps, sigmas, inv_T_scale=1000.0, return_for_plotting=True, return_err=True, w0=conv_factor)
    sigs.append(y)
    print(sigmas_err.shape)
    eas[k] = -slope * kB * 1e6 # in meV
    errs_lr[k] = err_lr * kB * 1e6 #error in Ea estimate from `arrhenius_fit`, in meV

    print(f'\n*** {cl} ***')
    print(f'linregress ---> {eas[k]} ± {errs_lr[k]} meV; sigma_0 = {np.exp(intercept)} [S/m] ; slope err = {interr_lr}')
    # print(np.sort(dcrits)[[0,-1]])
    # print(np.max(sigmas_err))


    if k < 3:
        ax.plot(x,np.exp(y),'o',label=cl,ms=10.0, c=cc)
    else:
        ax.plot(x,np.exp(y),'+',label=cl,ms=10.0, c=cc)
    # ax.errorbar(1000/temps,sigmas,yerr=sigmas_err,fmt='-o',c=cc,label=cl,ms=5.0)
    if k < 3:
        ax.plot(x, np.exp(x*slope+intercept),'-',c=cc,lw=1.6)
    else:
        ax.plot(x, np.exp(x*slope+intercept),'--',c=cc,lw=1.6)
    
    if k >= 3:
        print(f'\n--- Avg ratio between conductivities from |psi| and |psi|^2 sites = {np.mean(sigmas / sigs[k-3])}\n ---')

  
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
