#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico import plt_utils
from glob import glob




vdos_datadir = "/Users/nico/Desktop/simulation_outputs/MAC_MD_lammps/40x40/vdos/"
dfs = glob(vdos_datadir + "*.npy")
temps = np.array([int(df.split('-')[-1].split('.')[0][:-1]) for df in dfs])
clrs = plt_utils.get_cm(temps,cmap_str='coolwarm',max_val=1.0)

plt_utils.setup_tex()

for df,clr,T in zip(dfs,clrs,temps):
    if T in [np.min(temps), np.max(temps)]:
        dat = np.load(df)
        print(dat.shape)
        freqs, vdos = dat
        plt.plot(freqs,vdos,ls='-',lw=0.9,c=clr,label=f'$T = {T}\,$K')
    else:
        continue
#plt.colorbar()
plt.legend()
plt.show()