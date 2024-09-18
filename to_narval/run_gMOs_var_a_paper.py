#!/usr/bin/env python

import sys
from os import path
import numpy as np
from var_a_percolate_mac.deploy_percolate import load_data
from rollback_percolate_mac.deploy_percolate import run_var_a


"""
This version of the script produced the results that are currently in the manuscript.
Aug 8, 2024
"""


n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_type = sys.argv[3]

temps = np.arange(40,440,10)
dV = 0.0
rmax = 121.2
rho_min = 1e-4

npydir = 'sites_data_0.00105/'
run_name = f'rmax_{rmax}'

pos, e, M, gamL, gamR = load_data(n, struc_type, mo_type, compute_gammas=False)
run_var_a(pos,M,gamL,gamR,temps, dV, eF=0,hyperlocal=False,npydir=npydir,run_name=run_name,rmax=rmax,use_idprev=False)
