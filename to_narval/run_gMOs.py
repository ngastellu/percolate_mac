#!/usr/bin/env python

import sys
from os import path
import numpy as np
from percolate_mac.deploy_percolate import load_data, run_gridMOs


n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_type = sys.argv[3]

temps = np.arange(40,440,10)
dV = 0.0

pos, e, M, gamL, gamR = load_data(n, struc_type, mo_type, compute_gammas=True)
run_gridMOs(pos,e,M,gamL,gamR,temps, dV, eF=0)