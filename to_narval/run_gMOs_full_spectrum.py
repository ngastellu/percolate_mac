#!/usr/bin/env python

import sys
from os import path
import numpy as np
from time import perf_counter
from deploy_percolate import load_data, run_percolate, setup_hopping_sites_gridMOs


n = sys.argv[1]
struc_type = '20x20'

temps = np.arange(40,440,10)
dV = 0.0


pos, e, M, gamL, gamR = load_data(n, struc_type, compute_gammas=False,run_location='narval',full_spectrum=True)
N = pos.shape[0]
eHOMO = e[N//2 -1]
eLUMO = e[N//2]
eF = 0.5 * (eHOMO + eLUMO)

cc, ee, L, R = setup_hopping_sites_gridMOs(pos,e,M,gamL,gamR,nbins=100)

print('Running jitted...')
start = perf_counter()
run_percolate(cc,ee,L,R,temps, dV, eF=eF, jitted=True)
end = perf_counter()
print(f'Done! [{end-start} seconds]')
