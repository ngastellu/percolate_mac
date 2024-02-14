#!/usr/bin/env python

import sys
import pickle
from os import path
import numpy as np
import qcnico.qchemMAC as qcm
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from percolate import diff_arrs, percolate


sample_index = int(sys.argv[1])


all_Ts = np.arange(40,440,10)
mo_type = 'lo'

kB = 8.617e-5
rCC = 1.8
gamma = 0.1
dV = 0.0 #voltage drop across sample

# ******* 1: Load data *******
arpackdir = path.expanduser('~/scratch/ArpackMAC/40x40')
MOdir = path.join(arpackdir, f'MOs/{mo_type}/')
edir = path.join(arpackdir, f'energies/{mo_type}/')
pos_dir = path.expanduser('~/scratch/clean_bigMAC/40x40/relax/no_PBC/relaxed_structures')


mo_file = f'MOs_ARPACK_{mo_type}_bigMAC-{sample_index}.npy'
energy_file = f'eARPACK_{mo_type}_bigMAC-{sample_index}.npy'

mo_path = path.join(MOdir,mo_file)
energy_path = path.join(edir,energy_file)
pos_path = path.join(pos_dir,f'bigMAC-{sample_index}_relaxed.xsf')

energies = np.load(energy_path)
M =  np.load(mo_path)
pos, _ = read_xsf(pos_path)  
pos = remove_dangling_carbons(pos,rCC)


# ******* 2: Get gammas ******* 
agamL, agamR = qcm.AO_gammas(pos,gamma)
gamL, gamR = qcm.MO_gammas(M,agamL,agamR,return_diag=True)


# ******* 3: Define strongly-coupled MOs *******
tolscal = 3.0
gamL_tol = np.mean(gamL) + tolscal*np.std(gamL)
gamR_tol = np.mean(gamR) + tolscal*np.std(gamR)

L = set((gamL > gamL_tol).nonzero()[0])
R = set((gamR > gamR_tol).nonzero()[0])


# ******* 4: Pre-compute distances *******
# centres, ee, ii = generate_site_list(pos,M,L,R,energies,nbins=100)
# np.save('cc.npy',centres)
# np.save('ee.npy',ee)
# np.save('ii.npy', ii)
centres = qcm.MO_com(pos,M)
if np.abs(dV) > 0:
    dX = np.max(centres[:,0]) - np.min(centres[:,0])
    E = np.array([dV/dX,0])
else:
    E = 0
edArr, rdArr = diff_arrs(energies, centres, a0=30, eF=0, E=E)


for T in all_Ts:
    # ******* 5: Get spanning cluster *******
    conduction_clusters, dcrit, A = percolate(energies, pos, M, T, gamL_tol=gamL_tol,gamR_tol=gamR_tol, return_adjmat=True, distance='logMA',MOgams=(gamL, gamR), dArrs=(edArr,rdArr))

    with open(f'out_percolate-{T}K.pkl', 'wb') as fo:
        pickle.dump((conduction_clusters,dcrit,A), fo)
