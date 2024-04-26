#!/usr/bin/env python

import sys
from os import path
import numpy as np
import qcnico.qchemMAC as qcm
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from rollback_percolate_mac.deploy_percolate import run_gridMOs, load_data



n = int(sys.argv[1])
structype='tempdot6'
motype='virtual'

temps = np.arange(40,440,10)
dV = 0.0

pos, e, M, gamL, gamR = load_data(n, structype, motype,  compute_gammas=False)
run_gridMOs(pos,e,M,gamL,gamR,temps, dV, eF=0, compute_centres=False)
