#!/usr/bin/env python 

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from qcnico import plt_utils


def gen_data(lbls, ddir):
    for n in lbls:
        npy = ddir + f'sample-{n}/rr_v_masses_v_ee.npy'
        yield np.load(npy)

def gen_n_sites(lbls,ddir):
    for n in lbls:
        sdir = ddir + f'sample-{n}/'
        radii = np.load(sdir + 'radii.npy')
        nradii = radii.shape[0]
        ii = np.load(sdir + 'ii.npy')
        iunique, counts = np.unique(ii,return_counts=True)
        out = np.zeros((nradii,2))
        for i in range(nradii):
            iMO = ii[i]
            j = (iunique == iMO).nonzero()[0]
            print(j)
            nsites = counts[j]
            out[i,0] = radii[i]
            out[i,1] = nsites
        yield out






ddir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/var_radii_data/'
lbls = [int(d.split('-')[-1]) for d in  glob(ddir + 'sample-*')]


plt_utils.setup_tex()

fig, ax = plt.subplots()

for data in gen_n_sites(lbls,ddir):
    radii, counts = data.T
    ax.scatter(radii,counts, s=1.0,c='r')

# for data in gen_data(lbls, ddir):
#     rr, m, ee = data

#     ee -= np.min(ee)

#     ax.scatter(rr,m,c=ee,s=1.0)

ax.set_xlabel('Site radius [\AA]')
ax.set_ylabel('\# of sites')
# ax.set_ylabel('Mass')



plt.show()


