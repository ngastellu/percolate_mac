#!/usr/bin/env python

import sys
from os import path
import numpy as np
from var_a_percolate_mac.deploy_percolate import load_data
from rollback_percolate_mac.deploy_percolate import run_var_a


n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_type = sys.argv[3]

temps = np.arange(40,440,10)
dV = 0.0
rmax = 121.2

npydir = 'sites_data_0.00105_psi_pow1/'
run_name = f'rmax_{rmax}_psipow1'

pos, e, M, gamL, gamR = load_data(n, struc_type, mo_type, compute_gammas=False)
run_var_a(pos,M,gamL,gamR,temps, dV, eF=0,hyperlocal=False,npydir=npydir,run_name=run_name,use_idprev=False,rmax=rmax, check_sites=True)
