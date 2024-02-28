#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico import plt_utils
from glob import glob

Hz2invcm = 1.0/(2.9979e10)


vdos_datadir = "/Users/nico/Desktop/simulation_outputs/MAC_MD_lammps/4x4/vdos/"
# dfs = glob(vdos_datadir + "vibdos*.npy")
# temps = np.array([int(df.split('-')[-1].split('.')[0][:-1]) for df in dfs])
temps = [300,400]
dfs = [vdos_datadir + f'vibdos-{T}K.npy' for T in temps]
clrs = plt_utils.get_cm(temps,cmap_str='coolwarm',max_val=1.0)

plt_utils.setup_tex()

fig, ax = plt.subplots()

for df,clr,T in zip(dfs,clrs,temps):
    dat = np.load(df)
    print(dat.shape)
    freqs, vdos = dat
    ax.plot(freqs*Hz2invcm,vdos,ls='-',lw=0.9,c=clr,label=f'$T = {T}\,$K')


ax.set_xlabel('$\omega$ [cm$^{-1}$]')
ax.set_ylabel('VDOS (normalised to unity)')
# plt.colorbar()
plt.legend()
plt.show()
