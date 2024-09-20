#!/usr/bin/env python

import sys
from os import path
import numpy as np
from rollback_percolate_mac.deploy_percolate import run_locMOs, load_data


n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_type = sys.argv[3]


temps = np.arange(40,440,10)
dV = 0.0

run_name = f'{mo_type}_loc_var_a'
output_dir = f'{run_name}_pkls'

pos, e, M, gamL, gamR = load_data(n, struc_type, mo_type, gammas_method='load')
run_locMOs(pos,e,M,gamL,gamR,temps, dV, eF=np.min(e),var_a=True,run_name=run_name,pkl_dir=output_dir)
