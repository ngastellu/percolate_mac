#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt



def fermi_dirac(e,T):
    kB = 8.617333262e-5
    return 1.0 / (1+ np.exp(e/(kB*T)))

T = 300

# efiles = ['../full_device_test/2d/init/energies.npy', '../full_device_test/3d/init/energies.npy']
# pfiles =  ['../full_device_test/2d/init/prob.npy', '../full_device_test/3d/init/prob.npy']

# energies = np.hstack([np.load(ef) for ef in efiles])
# probs = np.hstack([np.load(pf) for pf in pfiles])

energies = np.load('../full_device_test/3d/init/energies.npy')[800:7200]
probs = np.load('../full_device_test/3d/init/prob.npy')[800:7200]

emin, emax = np.sort(energies)[[0,-1]]

p_fd = fermi_dirac(energies, T)

plt.plot(p_fd - probs,'ro')
plt.show()
