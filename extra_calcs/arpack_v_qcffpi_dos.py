#!/usr/bin/env pythonw

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import histogram, setup_tex

"""This script will compare the density of states of MAC obtained using: 
    * the ArpackMAC code run on 40nm x 40nm structures
    * the ArpackMAC code run on 10nm x 10nm structures
    * the QCFFPI code run on 10nm x 10nm structures"""

#MAC comparison
#with open('qcffpi_dos/orb_energy_pCNN_MAC_160x160.dat') as fo:
#with open('orb_energy_MAC_kMC_clean.dat') as fo:

datadir_arpack_40x40 = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/eARPACK/virtual/'
energies_arpack_40x40 = np.concatenate([np.load(npy) for npy in glob(datadir_arpack_40x40 + '*.npy')])

datadir_arpack_10x10 = '/Users/nico/Desktop/simulation_outputs/percolation/10x10/eARPACK/'
energies_arpack_10x10 = np.concatenate([np.load(npy) for npy in glob(datadir_arpack_10x10 + '*.npy')])

# datadir_qcffpi = 


setup_tex()

fig, ax = plt.subplots()

histogram(energies_arpack_10x10,normalised=True,xlabel='$E$ [eV]',plt_objs=(fig,ax),plt_kwargs={'color': 'b', 'alpha':0.5, 'label': '10$\\times$10 (ARPACK)'},show=False)
histogram(energies_arpack_40x40,normalised=True,xlabel='$E$ [eV]',plt_objs=(fig,ax),plt_kwargs={'color': 'r', 'alpha':0.5, 'label': '40$\\times$40 (ARPACK)'}, show=False)

plt.legend()
plt.show()
