#!/usr/bin/env python

import sys
import os
import numpy as np
from percolate_mac.deploy_percolate import load_data, run_percolate


n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_type = sys.argv[3]

temps = np.arange(180,440,10)
#temps = [300]
dV = 0.0

pos, e, M, gamL, gamR = load_data(n, struc_type, mo_type, compute_gammas=True)
efermi = 0.5 * (e[0] + e[1])

print(pos.shape)
pos = pos[:,:2]
gamL = gamL[1:]
gamR = gamR[1:]

tolscal = 3.0
L = set((gamL > (np.mean(gamL) + tolscal*np.std(gamL))).nonzero()[0])
R = set((gamR > (np.mean(gamR) + tolscal*np.std(gamR))).nonzero()[0])

pkl_dir = f'oldschool_{mo_type}'
if not os.path.isdir(pkl_dir):
    os.mkdir(pkl_dir)
run_percolate(pos,e[1:],L,R,temps, dV, eF=efermi)
