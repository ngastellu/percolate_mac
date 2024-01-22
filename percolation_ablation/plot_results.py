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
datadir = 'data_100x100_gridMOs_0V'
nsamples = np.hstack((np.arange(10,120,5), [116]))
sigmas = np.zeros((nsamples.shape[0], 40))
Eas = np.zeros(nsamples.shape[0])

clrs = plt_utils.get_cm(nsamples,cmap_str='inferno')

for k, n in enumerate(nsamples):
    T, sig = np.load(datadir+'/'+f'sigma_v_T-{n}samples.npy')
    sigmas[k,:] = sig
    Eas[k] = get_activation_energy(T,sig)


plt_utils.setup_tex(fontsize=20.0)


# fig, ax = plt.subplots()

# for n, c, sigs in zip(nsamples, clrs,sigmas):
#     ax.plot(1000.0/T, np.log(sigs),'-',c=c,lw=0.8, label=f"{n} samples")

# ax.set_xlabel("$1000/T$ [K]")
# ax.set_ylabel("$\log\sigma$")
# plt.legend()
# plt.legend()

fig, ax = plt.subplots()

ax.plot(nsamples, Eas, 'k-', lw=0.8)
ax.scatter(nsamples, Eas, c='k', s=4.8)
ax.set_xlabel("\# of samples")
ax.set_ylabel("Activation of energy [meV]")
plt.show()

print(f"Full data activation energy: {Eas[k]} meV")