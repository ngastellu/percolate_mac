#!/usr/bin/env python

import sys
from os import path
import numpy as np
from percolate_mac.deploy_percolate import load_data, run_percolate


n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_type = sys.argv[3]

# temps = np.arange(40,440,10)
temps = [300]
dV = 0.0

pos, e, M, gamL, gamR = load_data(n, struc_type, mo_type, compute_gammas=True)
efermi = 0.5 * (e[0] + e[1])
run_percolate(pos,e[1:],M[:,1:],gamL,gamR,temps, dV, eF=efermi)