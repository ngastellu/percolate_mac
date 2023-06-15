#!/usr/bin/env python

import sys
import pickle
from os import path
import numpy as np
import qcnico.qchemMAC as qcm
from qcnico.coords_io import read_xsf
from percolate import diff_arrs, percolate, get_MO_loc_centers


sample_index = int(sys.argv[1])




all_Ts = np.arange(40,440,10)

kB = 8.617e-5

# ******* 1: Load data *******
arpackdir = path.expanduser('~/scratch/ArpackMAC/40x40')
pos_dir = path.expanduser('~/scratch/clean_bigMAC/40x40/relax/relaxed_structures')
sample_dir = f'sample-{sample_index}'


mo_file = f'MOs_ARPACK_bigMAC-{sample_index}.npy'
energy_file = f'eARPACK_bigMAC-{sample_index}.npy'

mo_path = path.join(arpackdir,sample_dir,mo_file)
energy_path = path.join(arpackdir,sample_dir,energy_file)
pos_path = path.join(pos_dir,f'bigMAC-{sample_index}_relaxed.xsf')

energies = np.load(energy_path)
M =  np.load(mo_path)
pos, _ = read_xsf(pos_path)  


# ******* 2: Get gammas *******
gamL = np.load(f'gamL_40x40-{sample_index}.npy')
gamR = np.load(f'gamR_40x40-{sample_index}.npy')


# ******* 3: Define strongly-coupled MOs *******
tolscal = 3.0
gamL_tol = np.mean(gamL) + tolscal*np.std(gamL)
gamR_tol = np.mean(gamR) + tolscal*np.std(gamR)

biggaL_inds = (gamL > gamL_tol).nonzero()[0]
biggaR_inds = (gamR > gamR_tol).nonzero()[0]


# ******* 4: Pre-compute distances *******
centres = np.zeros(2)
ee = []
for n in range(len(energies)):
    cc =get_MO_loc_centers(pos,M,n)
    print(len(cc))
    centres = np.vstack([centres,cc])
    ee.extend([energies[n]]*cc.shape[0])
ee = np.array(ee)
centres = centres[1:,:]
np.save('cc.npy',centres)
np.save('ee.npy',ee)
edArr, rdArr = diff_arrs(ee, centres, a0=30, eF=0)

for T in all_Ts:
    # ******* 5: Get spanning cluster *******
    conduction_clusters, dcrit, A = percolate(ee, pos, M, T, gamL_tol=gamL_tol,gamR_tol=gamR_tol, return_adjmat=True, distance='logMA',MOgams=(gamL, gamR), dArrs=(edArr,rdArr))

    with open(f'out_percolate-{T}K.pkl', 'wb') as fo:
        pickle.dump((conduction_clusters,dcrit,A), fo)
