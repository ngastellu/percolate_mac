#!/usr/bin env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex



E = np.linspace(40000, 2000000, 10000)
mu = np.load('mu_1d.npy')


setup_tex()

plt.plot(np.sqrt(E), np.log10(mu), 'r-', lw=0.8)
plt.xlabel('$\sqrt{E}$ [V$^{1/2}$/cm${1/2}$]')
plt.ylabel('$\\text{log}_{10}\,\mu$ [cm$^{2}$/Vs]')
plt.show()