#!/usr/bin/env python

import sys
from os import path
import numpy as np
from rollback_percolate_mac.deploy_percolate import run_locMOs, load_data


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

run_name = f'rmax_{rmax}_psipow{psipow}_locMOs_var_a'

pos, e, M, gamL, gamR = load_data(n, struc_type, mo_type, compute_gammas='load')
run_locMOs(pos,M,gamL,gamR,temps, dV, eF=np.min(e),run_name=run_name,use_idprev=False,rmax=rmax, check_sites=True)
