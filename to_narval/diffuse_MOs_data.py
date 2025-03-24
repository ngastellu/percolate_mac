#!/usr/bin/env python

import sys
from os import path
from glob import glob
import numpy as np


Mdir = path.expanduser(sys.argv[1])
structype = sys.argv[2]
run_files = glob(Mdir + '/*npy')


psi_maxs = np.zeros(len(run_files))
max_p_low = np.zeros(len(run_files))
ndiffuse_MOs = np.zeros(len(run_files),dtype='int')

threshold = 0.001


for k, npy in enumerate(run_files):
    print(k)
    M = np.abs(np.load(npy))**2
    psi_maxs[k] = np.min(np.max(M,axis=0)) # we're interested in the smallest max; we want to see if any MOs will be completely nixed by thresholding
    Mbool = (M < threshold)
    p_lows = 100 * Mbool.sum(0) / M.shape[0]
    max_p_low[k] = np.max(p_lows) 
    ndiffuse_MOs[k] = (p_lows == 100).sum()

np.save(f'min_psi_max_{structype}.npy', psi_maxs)
np.save(f'max_p_low_{structype}.npy', p_lows)
np.save(f'ndiffuse_MOs_{structype}.npy', ndiffuse_MOs)