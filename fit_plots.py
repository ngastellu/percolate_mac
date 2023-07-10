#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import linregress
from qcnico import plt_utils


first_ind = 14
temps, sigmas = np.load('sigma_v_T.npy')[:,first_ind:]
y = np.log(sigmas)

print(temps[0])
print(temps[-1])


plt_utils.setup_tex()
rcParams['font.size'] = 20
rcParams['figure.dpi'] = 150.0 #increasre figure resolution
rcParams['figure.subplot.top'] = 0.92
rcParams['figure.subplot.bottom'] = 0.17
#rcParams['figure.figsize'] = [3.2,2.4]

#fig, axs = plt.subplots(3,1)

# Do Mott linear regression d = 0
x = 1000.0/temps
slope, intercept, r, *_ = linregress(x,y)
print(f'0D Mott VRH: slope = {slope}, r^2 = {r**2}')
# axs[0].plot(x,y,'ko',ms=2.0)
# axs[0].plot(x, x*slope + intercept,'r-')
# axs[0].set_xlabel("$1000/T$ [K$^{-1}$]")
# axs[0].set_ylabel("$\log \sigma$")
plt.plot(x,y,'ko',ms=2.0)
plt.plot(x, x*slope + intercept,'r-')
plt.xlabel("$1000/T$ [K$^{-1}$]")
plt.ylabel("$\log \sigma$")
plt.show()
#ax.set_yscale('log')
#axs[0].set_title(f'Mott $d=0$ [$r^2 = $ {r**2}]')

kB = 8.617e-5

print(slope*kB*1000)


# Do Mott linear regression d = 2
x = np.power(1.0/temps,3)
slope, intercept, r, *_ = linregress(x, y)
print(f'2D Mott VRH: slope = {slope}, r^2 = {r**2}')
# axs[1].plot(x,y,'ko',ms=2.0)
# axs[1].plot(x,intercept + slope*x,'r-',lw=0.8)
# axs[1].set_xlabel("$T^{-1/3}$ [K$^{-1/3}$]")
# axs[1].set_ylabel("$\log \sigma$")
plt.plot(x,y,'ko',ms=2.0)
plt.plot(x,intercept + slope*x,'r-',lw=0.8)
plt.xlabel("$T^{-1/3}$ [K$^{-1/3}$]")
plt.ylabel("$\log \sigma$")
plt.show()
#ax.set_yscale('log')
#axs[1].set_title(f'Mott $d=2$ [$r^2 = $ {r**2}]')


# Do RCH linear regression
x = np.log(temps)
slope, intercept, r, *_ = linregress(x, y)
print(f'RCH: slope = {slope}, r^2 = {r**2}')
# axs[2].plot(x,y,'ko',ms=2.0)
# axs[2].plot(x,intercept + slope*x,'r-',lw=0.8)
# axs[2].set_xlabel("$\log T$")
# axs[2].set_ylabel("$\log \sigma$")
plt.plot(x,y,'ko',ms=2.0)
plt.plot(x,intercept + slope*x,'r-',lw=0.8)
plt.xlabel("$\log T$")
plt.ylabel("$\log \sigma$")
#ax.set_yscale('log')
#axs[2].set_title(f'Rare chain hopping [$r^2 = $ {r**2}]')
plt.show()

# plt.tight_layout()
# plt.show()


