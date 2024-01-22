#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from qcnico import plt_utils


def get_activation_energy(Ts,sigmas):    
    kB = 8.617333262e-5
    
    x = 1000.0/Ts
    y = np.log(sigmas)
    slope, intercept, r, *_ = linregress(x,y)

    print("r^2 = ",r**2)

    return -slope*kB*1e6 #in meV




# datadir = 'data_100x100_gridMOs_1V'
# nsamples = np.hstack((np.arange(10,170,5), [173]))
datadir_0V = 'data_100x100_gridMOs_0V'
nsamples_0V = np.hstack((np.arange(10,120,5), [116]))
sigmas_0V = np.zeros((nsamples_0V.shape[0], 40))
Eas_0V = np.zeros(nsamples_0V.shape[0])

clrs = plt_utils.get_cm(nsamples_0V,cmap_str='inferno')

for k, n in enumerate(nsamples_0V):
    T, sig = np.load(datadir_0V+'/'+f'sigma_v_T-{n}samples.npy')
    sigmas_0V[k,:] = sig
    Eas_0V[k] = get_activation_energy(T,sig)


datadir_1V = 'data_100x100_gridMOs_1V'
nsamples_1V = np.hstack((np.arange(10,170,5), [173]))
sigmas_1V = np.zeros((nsamples_1V.shape[0], 40))
Eas_1V = np.zeros(nsamples_1V.shape[0])

clrs = plt_utils.get_cm(nsamples_1V,cmap_str='inferno')

for k, n in enumerate(nsamples_1V):
    T, sig = np.load(datadir_1V+'/'+f'sigma_v_T-{n}samples.npy')
    sigmas_1V[k,:] = sig
    Eas_1V[k] = get_activation_energy(T,sig)


plt_utils.setup_tex(fontsize=20.0)


# fig, ax = plt.subplots()

# for n, c, sigs in zip(nsamples, clrs,sigmas):
#     ax.plot(1000.0/T, np.log(sigs),'-',c=c,lw=0.8, label=f"{n} samples")

# ax.set_xlabel("$1000/T$ [K]")
# ax.set_ylabel("$\log\sigma$")
# plt.legend()
# plt.legend()

fig, ax = plt.subplots(2,1,sharex=True)

ax[0].plot(nsamples_0V, Eas_0V, 'k-', lw=0.8)
ax[0].scatter(nsamples_0V, Eas_0V, c='k', s=4.8)
ax[0].set_xlabel("\# of samples")
ax[0].set_ylabel("Activation of energy [meV]")

ax[1].plot(nsamples_1V, Eas_1V, 'k-', lw=0.8)
ax[1].scatter(nsamples_1V, Eas_1V, c='k', s=4.8)
ax[1].set_xlabel("\# of samples")
ax[1].set_ylabel("Activation of energy [meV]")

plt.show()

print(f"Full data activation energy: {Eas_1V[k]} meV")