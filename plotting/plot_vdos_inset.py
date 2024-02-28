#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico import plt_utils
from glob import glob
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                          mark_inset)




vdos_datadir = "/Users/nico/Desktop/simulation_outputs/MAC_MD_lammps/40x40/vdos/"
dfs = glob(vdos_datadir + "*.npy")
temps = np.array([int(df.split('-')[-1].split('.')[0][:-1]) for df in dfs])
clrs = plt_utils.get_cm(temps,cmap_str='coolwarm',max_val=1.0)

plt_utils.setup_tex()

fig, ax = plt.subplots()

# for df,clr,T in zip(dfs,clrs,temps):
#     dat = np.load(df)
#     print(dat.shape)
#     freqs, vdos = dat
#     ax.plot(freqs,vdos,ls='-',lw=0.9,c=clr,label=f'$T = {T}\,$K')

plot_ind = np.argsort(temps)[-1]
T = temps[plot_ind]
clr = clrs[plot_ind]
dat = np.load(dfs[plot_ind])
freqs, vdos = dat
ax.plot(freqs,vdos,ls='-',lw=0.9,c=clr)




# Inset
inset_inds = ((freqs > 0.014)*(freqs < 0.026)).nonzero()[0]
inset_position = [0.4,0.2,0.5,0.5]

ax2 = plt.axes([0,0,1,1])
ip = InsetPosition(ax, inset_position)
ax2.set_axes_locator(ip)
mark_inset(ax,ax2,loc1=2,loc2=3,fc='none',ec='k',lw=0.6,zorder=10)
ax2.plot(freqs[inset_inds],vdos[inset_inds],c=clr,ls='-',lw=0.8)

ax.set_xlabel('Frequency [fs$^{-1}$]')
ax.set_ylabel('VDOS')
#plt.colorbar()
#plt.legend()
plt.show()
