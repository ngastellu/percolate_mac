#!/usr/bin/env python

import sys
from os import path
import numpy as np
from var_a_percolate_mac.deploy_percolate import load_data
from rollback_percolate_mac import run_var_a


n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_type = sys.argv[3]

temps = np.arange(40,440,10)
dV = 0.0

pos, e, M, gamL, gamR = load_data(n, struc_type, mo_type, compute_gammas=False)
run_var_a(pos,M,gamL,gamR,temps, dV, eF=0)