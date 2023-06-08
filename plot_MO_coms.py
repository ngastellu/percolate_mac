#!/usr/bin/env python

from os import path
import numpy as np
import matplotlib.pyplot as plt
from qcnico.qchemMAC import MO_com, AO_gammas, MO_gammas
from qcnico.coords_io import read_xsf
from qcnico.qcplots import plot_atoms
from qcnico.plt_utils import setup_tex, get_cm

# ***** 1: Load data *****

datadir = path.expanduser('~/Desktop/simulation_outputs/percolation/10x10')
energy_dir = 'eARPACK'
mo_dir = 'MOs_ARPACK'
pos_dir = 'structures'


sample_index = 2
mo_file = f'MOs_ARPACK_bigMAC-{sample_index}.npy'
energy_file = f'eARPACK_bigMAC-{sample_index}.npy'

mo_path = path.join(datadir,mo_dir,mo_file)
energy_path = path.join(datadir,energy_dir,energy_file)
pos_path = path.join(datadir,pos_dir,f'bigMAC-{sample_index}_relaxed.xsf')

energies = np.load(energy_path)
M =  np.load(mo_path)
pos, _ = read_xsf(pos_path) 
pos = pos[:,:2] 
print(pos.shape)


# ***** 2: Compute lead-couplings *****

ga = 0.1 #edge atome-leaf coupling in eV
print("Computing AO gammas...")
agaL, agaR = AO_gammas(pos,ga)
print("Computing MO gammas...")
gamL, gamR = MO_gammas(M,agaL, agaR, return_diag=True)

# ***** 3: Define sets of left-, and right-coupled MOs, etc. *****

tolscal = 1.0

gamL_tol = np.mean(gamL) + tolscal*np.std(gamL)
gamR_tol = np.mean(gamR) + tolscal*np.std(gamR)

biggaL_inds = (gamL > gamL_tol).nonzero()[0]
biggaR_inds = (gamR > gamR_tol).nonzero()[0]

all_inds = set(range(M.shape[1]))
L = set([n for n in biggaL_inds])
R = set([n for n in biggaR_inds])
bulk_inds = all_inds - L - R

doublecouple_inds = L & R
if len(doublecouple_inds) > 0:
    print("Inds of MOs coupled to BOTH leads = ", doublecouple_inds)

# ***** 4: Plot results *****

setup_tex()

fig, ax = plt.subplots()

comsL = MO_com(pos,M,list(L))
comsR = MO_com(pos,M,list(R))
coms_both = MO_com(pos,M,list(doublecouple_inds))
coms_bulk = MO_com(pos,M,list(bulk_inds))
all_coms = MO_com(pos,M)

cm_inds = get_cm(np.arange(M.shape[1]),'rainbow',max_val=1)
cm_ee = get_cm(energies,'viridis')

print(comsL.shape)



plot_atoms(pos,dotsize=25,show=False,plt_objs=(fig,ax))
ax.scatter(*(comsL.T),marker='*',color='red',label='L',s=100)
ax.scatter(*(comsR.T),marker='*',color='dodgerblue',label='R',s=100)
ax.scatter(*(coms_bulk.T),marker='*',color='lime',label='bulk',s=100)
ax.scatter(*(coms_both.T),marker='*',color='fuchsia',label='both',s=100)
ax.scatter(*(all_coms.T),marker='o',s=300,edgecolor=cm_inds,c='none')
plt.legend()
plt.colorbar()
plt.show()
