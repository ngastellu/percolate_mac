#!/usr/bin/env python

import sys
from os import path
import numpy as np
from var_a_percolate_mac.deploy_percolate import load_data, load_var_a_data, run_percolate
from var_a_percolate_mac.MOs2sites import LR_MOs


n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_type = sys.argv[3]

temps = np.arange(40,440,10)
dV = 0.0

pos, e, M, gamL, gamR = load_data(n, struc_type, mo_type, compute_gammas=False)
L, R = LR_MOs(gamL, gamR)
sites_pos, sites_energies, sites_radii, ii = load_var_a_data(datadir='.')
run_percolate(sites_pos,sites_energies,L,R,temps, dV, a0=sites_radii, eF=0, jitted=False)