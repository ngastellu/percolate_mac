#!/usr/bin/env python

import sys
import os
import numpy as np
from rollback_percolate_mac.deploy_percolate import run_locMOs, load_data



n = int(sys.argv[1])
mo_type = sys.argv[2]

struc_type = os.path.basename(os.getcwd())

if struc_type == '40x40':
    rmax = 18.03
elif struc_type == 'tempdot6':
    rmax = 121.2
elif struc_type == 'tempdot5':
    rmax = 198.69
else:
    print(f'Structure type {struc_type} is invalid. Exiting in anger.')
    sys.exit()


temps = np.arange(40,440,10)
dV = 0.0

run_name = f'{mo_type}_loc_var_a_rmax_{rmax}'
output_dir = f'{run_name}_pkls'

pos, e, M, gamL, gamR = load_data(n, struc_type, mo_type, gammas_method='load')
run_locMOs(pos,e,M,gamL,gamR,temps, dV, eF=np.min(e),var_a=True,run_name=run_name,pkl_dir=output_dir,rmax=rmax)