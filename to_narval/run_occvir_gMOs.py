#!/usr/bin/env python

import sys
from os import path
import numpy as np
from percolate_mac.deploy_percolate import load_data_multi, run_gridMOs


n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_types = ['occupied', 'virtual']



temps = np.arange(40,440,10)
dV = 0.0

pos, e, M, gamL, gamR = load_data_multi(n, struc_type, mo_types, compute_gammas=True)
run_gridMOs(pos,e,M,gamL,gamR,temps, dV, eF=0)