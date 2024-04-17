#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from qcnico.qcplots import plot_MO



M = np.load('/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tempdot6/MOs_ARPACK/virtual/MOs_ARPACK_bigMAC-127.npy')
pos, _ = read_xsf('/Users/nico/Desktop/simulation_outputs/MAC_structures/Ata_structures/tempdot6/relaxed/tempdot6n127_relaxed.xsf')

percdir_127 = '/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tempdot6/percolate_output/zero_field/virt_100x100_gridMOs/sample-127/'
old_dir = percdir_127 + 'old_sites/'
numba_dir = percdir_127 + 'numba_opt_sites/'



pos = remove_dangling_carbons(pos,1.8)
N = pos.shape[0]
nMOs= M.shape[1]

iold = np.load(old_dir + 'ii.npy')
inumba = np.load(numba_dir + 'ii.npy')

old_sites = np.load(old_dir + 'cc.npy')
numba_sites = np.load(numba_dir + 'cc.npy')

old_energies = np.load(old_dir + 'ee.npy')
numba_energies = np.load(numba_dir + 'ee.npy')

for i in range(110 ,nMOs):

    iold_target_MO= (iold == i).nonzero()[0]
    inumba_target_MO= (inumba == i).nonzero()[0]
    print(f'\n**** {i} ****')
    print('Old energies all the same = ', np.all(old_energies[iold_target_MO] == old_energies[iold_target_MO][0]))
    print('Numba energies all the same = ', np.all(numba_energies[inumba_target_MO] == numba_energies[inumba_target_MO][0]))
    print('Numba and old target_MO energies match = ', old_energies[0] == numba_energies[0])


    old_target_MO_sites = old_sites[iold_target_MO]
    numba_target_MO_sites = numba_sites[inumba_target_MO]

    print(old_target_MO_sites.shape)

    try:
        cmp = np.all(old_target_MO_sites == numba_target_MO_sites,axis=1)
    except ValueError as e:
        print(e)
        cmp = np.array([False])
        print(e)
        print(old_target_MO_sites[:,0].shape)
        print(numba_target_MO_sites[:,0].shape)

    if np.all(cmp):
        print('All sites match!')
    else:
        print(f'Non-matching sites [{np.sum(~cmp)} of {cmp.shape[0]}] = ', (~cmp).nonzero()[0])
        break

fig, ax = plot_MO(pos, M,i,dotsize=1,show=False)

ax.scatter(x=old_target_MO_sites[:,0], y=old_target_MO_sites[:,1], c='r',marker='*',s=50)
ax.scatter(x=numba_target_MO_sites[:,0], y=numba_target_MO_sites[:,1], marker='*',c='limegreen',s=20)
plt.show()


