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
motype = 'virtual'

# temps = np.arange(40,440,10)[14:]

r_maxs = ['18.03', '121.2', '198.69']
curve_lbls = ['sAMC-500', 'sAMC-q400', 'sAMC-300']
structypes=['40x40', 'tempdot6', 'tempdot5']

clrs = MAC_ensemble_colours()

ndatasets = len(curve_lbls)

setup_tex()


rcParams['font.size'] = 25
# rcParams['figure.figsize'] = [8,7]

fig, ax = plt.subplots()

eas = np.zeros(ndatasets)
errs_lr = np.zeros(ndatasets)

eas2 = np.zeros(ndatasets)
errs_cf = np.zeros(ndatasets)

sigs = []


# for k, dd, ll, cc, cl in zip(range(2), ddirs, run_lbls, clrs, curve_lbls):
for k, st, rr, cc, cl in zip(range(ndatasets), structypes, r_maxs, clrs, curve_lbls):


    # print(f'~~~~~~~~ Color = {cc} ~~~~~~~~~')

    #run_name = f'rmax_{rr}_psipow2_sites_gammas_{motype}''
    run_name = f'rmax_{rr}_sites_gammas_{motype}'
    temps, sigmas, sigmas_err = np.load(path.join(sigmasdir, f'sigma_v_T_w_err_{st}_{run_name}.npy')).T

    # if k == 1:
    temps = temps[14:]
    sigmas = sigmas[14:]
    sigmas_err = sigmas_err[14:] * conv_factor / (kB*temps)

    # sigs.append(sigmas)
 
    slope, intercept, x, y, err_lr, interr_lr  = arrhenius_fit(temps, sigmas, inv_T_scale=1000.0, return_for_plotting=True, return_err=True, w0=conv_factor)
    sigs.append(y)
    # print(sigmas_err.shape)
    eas[k] = -slope * kB * 1e6 # in meV
    errs_lr[k] = err_lr * kB * 1e6 #error in Ea estimate from `arrhenius_fit`, in meV

    print(f'\n*** {cl} ***')
    print(f'linregress ---> {eas[k]} ± {errs_lr[k]} meV; sigma_0 = {np.exp(intercept)} [S/m] ; slope err = {interr_lr}')
    # print(np.sort(dcrits)[[0,-1]])
    # print(np.max(sigmas_err))


    # ax.plot(x,np.exp(y),'o',label=cl,ms=7.0, c=cc)
    ax.errorbar(1000/temps,sigmas*conv_factor/(kB*temps),yerr=sigmas_err,fmt='o',c=cc,label=cl,ms=5.0)
    ax.plot(x, np.exp(x*slope+intercept),'-',c=cc,lw=1.2)
    


  
ax.set_yscale('log')
ax.set_xlabel('$1000/T$ [K$^{-1}$]')
ax.set_ylabel('$\sigma$ [S/m]')
# ax.set_title('Percolation using mid-band virtual MOs (with $k$-clustering and $r_{cut}$)')

plt.legend()
plt.show()


# for k, s in enumerate(sigs):
#     print(f'\n----------{k}----------')
#     print(s)

# print(np.exp(sigs[1])/np.exp(sigs[2]))
