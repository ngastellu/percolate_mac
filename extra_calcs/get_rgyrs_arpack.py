#!/usr/bin/env python

import numpy as np
from time import perf_counter
from qcnico.qchemMAC import MO_rgyr


for n in range(32):
    print(f'***** {n} *****')
    pos = np.load(f'coords/coords-{n}.npy')

    print('Doing occupied MOs...')
    start = perf_counter()
    Mocc = np.load(f'MOs/occupied/MOs_ARPACK_bigMAC-{n}.npy')
    rgyrs_occ = MO_rgyr(pos, Mocc)
    end = perf_counter()
    print(f'Done! [{end-start} s]')
    
    print('Doing virtual MOs...')
    start = perf_counter()
    Mvir = np.load(f'MOs/virtual/MOs_ARPACK_bigMAC-{n}.npy')
    rgyrs_vir = MO_rgyr(pos, Mvir)
    end = perf_counter()
    print(f'Done! [{end-start} s]')

    np.save(f'rgyrs/occupied/rgyrs_tempdot5_occ-{n}.npy', rgyrs_occ)
    np.save(f'rgyrs/virtual/rgyrs_tempdot5_vir-{n}.npy', rgyrs_vir)