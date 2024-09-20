#!/usr/bin/env python
import numpy as np
from var_a_percolate_mac.utils_analperc import get_dcrits, sigma_errorbar
import os
import sys



structype = sys.argv[1]
motype = sys.argv[2]
psipow = 2
temps = np.arange(180,440,10)

if structype == '40x40':
    rmax = 18.03
elif structype == 'tempdot6':
    rmax = 121.2
elif structype == 'tempdot5':
    rmax = 198.69


run_name = f'rmax_{rmax}_psipow{psipow}_sites_gammas_{motype}'

datadir = f'{run_name}_symlinks'
with open(os.path.join(datadir, f'good_runs_rmax_{rmax}_psipow{psipow}_sites_gammas_{motype}.txt')) as fo:
    nn = [int(l.strip()) for l in fo]


dcrits = get_dcrits(nn, temps, datadir, pkl_prefix=f'out_percolate_{run_name}')

sigmas, sigmas_err = sigma_errorbar(dcrits)

outdir = 'sigmas_v_T/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

np.save(os.path.join(outdir, f'sigma_v_T_w_err_{run_name}.npy'), np.vstack((temps, sigmas, sigmas_err)).T)
