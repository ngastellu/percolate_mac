#!/usr/bin/env python
import numpy as np
from utils_analysis import load_dcrits, sigma_errorbar
import os
import sys



structype = os.path.basename(os.getcwd())
motype = sys.argv[1]

if structype == 'sAMC-500':
    nn = np.arange(1,301)
else:
    nn = np.arange(218)



#run_name = f'rmax_{rmax}_psipow{psipow}_sites_gammas'
# run_name = f'production_{motype}'
run_name = f'production'

temps, dcrits = load_dcrits(nn, run_name)

sigmas, sigmas_err = sigma_errorbar(dcrits)

outdir = 'sigmas_v_T/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

np.save(os.path.join(outdir, f'sigma_v_T_w_err_{run_name}.npy'), np.vstack((temps, sigmas, sigmas_err)).T)
