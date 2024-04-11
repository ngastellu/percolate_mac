#!/usr/bin/env python

from os import path
import numpy as np
from time import perf_counter
from deploy_percolate import load_data, run_percolate, setup_hopping_sites_gridMOs


n = 300
struc_type = 'pCNN'
mo_type = 'virtual_cleaned'

temps = np.arange(100,150,10)
dV = 0.0


pos, e, M, gamL, gamR = load_data(n, struc_type, mo_type, compute_gammas=False,run_location='local')
cc, ee, L, R = setup_hopping_sites_gridMOs(pos,e,M,gamL,gamR,nbins=40)

print('Running jitted...')
start = perf_counter()
run_percolate(cc,ee,L,R,temps, dV, eF=np.min(e), jitted=True)
end = perf_counter()
print(f'Nugget! [{end-start} seconds]')

print('Running vanilla...')
start = perf_counter()
run_percolate(cc,ee,L,R,temps, dV, eF=np.min(e), jitted=False)
end = perf_counter()
print(f'Nugget 2! [{end-start} seconds]')