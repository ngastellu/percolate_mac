#!/usr/bin/env python

import sys
from os import path
import numpy as np
from var_a_percolate_mac.deploy_percolate import load_data
from var_a_percolate_mac.deploy_percolate import run_var_a_from_sites


n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_type = sys.argv[3]

# rmax = np.inf

if struc_type == '40x40':
  rmax = 18.03
elif struc_type == 'tempdot6' or struc_type == 'equil_tempdot6':
  rmax = 136.47
elif struc_type == 'tempdot5' or struc_type == 'equil_tempdot5':
  rmax = 199.33
else:
  print(f'Structure type {struc_type} is not a valid  entry. Exiting now.')
  sys.exit()

#temps = np.arange(40,440,10)
temps=np.array([300])
dV = 0.0
kB = 8.617333e-5
Tref = 300 # reference temperature used to determine which energies are thermally accessible to hopping
e_thermal = 4 * kB *Tref


npydir = f'sites_data_{mo_type}/'
# run_name = f'no_rmax_{mo_type}'
run_name = f'rerun_rmax_{rmax}_{mo_type}'
pkl_dir = f'{run_name}_pkls/'

pos, e, M = load_data(n, struc_type, mo_type, gammas_method='none',sort_energies=True)
S = np.load(npydir + 'site_state_matrix.npy')

if mo_type == 'lo' or mo_type == 'virtual':
   efermi = np.min(e)
elif mo_type == 'hi': 
    efermi = np.max(e)
else: # mo_type is 'virtual_w_HOMO': using mid-band states (we assume `e` is sorted here)
    N = pos.shape[0]
    if N % 2 == 0:
        eHOMO = e[0]
        eLUMO = e[1]
        efermi = 0.5 * (eHOMO + eLUMO)
    else:
       efermi = e[0]

run_var_a_from_sites(pos,M,S,temps, dV, eF=efermi,hyperlocal=False,npydir=npydir,run_name=run_name,rmax=rmax,dE_max=e_thermal,rotate=False,pkl_dir=pkl_dir)
