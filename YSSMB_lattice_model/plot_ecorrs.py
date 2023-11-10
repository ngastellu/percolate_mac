#!/usr/bin/env python 

from os import path
import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex


ddir = path.expanduser('~/Desktop/simulation_outputs/yssmb_hopping')

N1 = 32
N2 = 16

T = 300 # K
kB = 8.617333262e-5 # eV/K
nu = 0.3
K = 0.0034

nstring = f'{N1}x{N2}x{N2}'
pdists = np.load(path.join(ddir, f'pairdists_{nstring}.npy'))
ecorr_avg = np.load(path.join(ddir, f'avg_ecorr_{nstring}.npy'))

y = ((nu**2)*kB*T)/(4*np.pi*K*pdists)

setup_tex()

plt.plot(pdists, ecorr_avg,'r-', label='data')
plt.plot(pdists, y, 'k--', label='theory')
plt.xlabel('$R_{ij}$ [$\\text{\AA}$]')
plt.ylabel("$\langle\\varepsilon(\\bm{r}_i)\,\\varepsilon(\\bm{r}_j\\rangle$ [eV$^2$]")
plt.legend()
plt.show()

