#!/usr/bin/env python

import sys
from os import path
import numpy as np
from var_a_percolate_mac.deploy_percolate import load_data
from rollback_percolate_mac.deploy_percolate import run_var_a_from_sites


n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_type = sys.argv[3]
psipow = int(sys.argv[4])

if struc_type == '40x40':
    rmax = 18.03
elif struc_type == 'tempdot6':
    rmax = 121.2
elif struc_type == 'tempdot5':
    rmax = 198.69
else:
    print(f'Structure type {struc_type} is not a valid  entry. Exiting now.')
    sys.exit()

temps = np.arange(40,440,10)
dV = 0.0

if mo_type == 'virtual':
    npydir = f'sites_data_0.00105_psi_pow{psipow}/'
    run_name = f'rmax_{rmax}_psipow{psipow}_sites_gammas_rotated'
else:
    npydir = f'sites_data_0.00105_psi_pow{psipow}_{mo_type}/'
    run_name = f'rmax_{rmax}_psipow{psipow}_sites_gammas_{mo_type}_rotated'

pos, e, M = load_data(n, struc_type, mo_type, gammas_method='none')
S = np.load(npydir + 'site_state_matrix.npy')
run_var_a_from_sites(pos,M,S,temps, dV, eF=0,hyperlocal=False,npydir=npydir,run_name=run_name,rmax=rmax,rotate=True)
