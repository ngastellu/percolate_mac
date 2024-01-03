#!/usr/bin env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex



E = np.linspace(40000, 2000000, 10000) * 1e-8
mu = np.load('mu_1d.npy')
# E, mu = np.load('/Users/nico/Desktop/simulation_outputs/yssmb_hopping/varE_16x8x8/task-4_smol_data.npy')


setup_tex()

plt.plot(np.sqrt(E), np.log10(mu), 'r-', lw=0.8)
plt.xlabel('$\sqrt{E}$ [V$^{1/2}$/cm${1/2}$]')
plt.ylabel('$\\text{log}_{10}\,\mu$ [cm$^{2}$/Vs]')
plt.suptitle('1D lattice $1024$ sites')
plt.show()