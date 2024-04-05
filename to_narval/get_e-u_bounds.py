#!/usr/bin/env python

import sys
from os import path
import numpy as np
from scipy.linalg import eigh
from percolate import diff_arrs
from MOs2sites import generate_site_list
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from qcnico import qchemMAC as qcm


nn = int(sys.argv[1])

kB = 8.617e-5 #eV/K
rCC = 1.8 #angstroms
gamma = 0.1 #eV
tolscal_gamma = 3.0

Tmin = 40
Tmax = 430

datadir = path.expanduser('~/Desktop/simulation_outputs/percolation/10x10/')
arpackdir = path.join(datadir,'Hao_ARPACK')
strucdir = path.join(datadir, 'structures')

# Get atomic positions
print('Loading pos...')
pos, _ = read_xsf(path.join(strucdir,f'bigMAC-{nn}_relaxed.xsf'))
print('Done!')
print('Removing dangling Cs...')
pos = remove_dangling_carbons(pos,rCC)
print('Done!\n')
N = pos.shape[0]

print('Loading H...')
hvals = np.load(path.join(arpackdir,'hvals', f'hvals-{nn}.npy'))
ii = np.load(path.join(arpackdir,'inds', f'ii-{nn}.npy'))
jj = np.load(path.join(arpackdir,'inds', f'jj-{nn}.npy'))
H = np.zeros((N,N))
H[ii-1,jj-1] = hvals
H += H.T
print('Done!\n')

print('Diagonalising...')
energies, M = eigh(H) 
print('Done!\n')
# np.save(f'extreme_energies-{nn}.npy', np.array([np.min(energies), np.max(energies)]))

eF = 0.5 * (energies[N//2 -1] + energies[N//2])
energies -= eF

print('Getting gammas...')
agL, agR = qcm.AO_gammas(pos, gamma)
gamL, gamR = qcm.MO_gammas(pos, agL, agR, return_diag=True)
# np.save(f'gamL_40x40-{nn}.npy', gamL)
# np.save(f'gamR_40x40-{nn}.npy', gamR)

# ******* Define strongly-coupled MOs *******
gamL_tol = np.mean(gamL) + tolscal_gamma*np.std(gamL)
gamR_tol = np.mean(gamR) + tolscal_gamma*np.std(gamR)

L = set((gamL > gamL_tol).nonzero()[0])
R = set((gamR > gamR_tol).nonzero()[0])
print('Done!\n')

print('Getting centres...')
centres, ee, ii = generate_site_list(pos,M,L,R,energies,nbins=100)
# np.save(f'cc.npy',centres)
# np.save(f'ee.npy',ee)
# np.save(f'ii.npy', ii)    
print('Done!\n')

print('Computing difference arrays...')
dE, dR = diff_arrs(ee, centres, eF=eF, a0=30,E=np.array([0.0,0.0]))
print('Done!\n')

print('Getting max/min dists...')
dsave = np.zeros(2)
dists = (dE/(kB*Tmin)) + dR
dsave[0] = np.max(dists)

dists = (dE/(kB*Tmax)) + dR
dsave[0] = np.min(dists)

np.save(f'extreme_dists-{nn}.npy', dsave)
print('Done!\n')