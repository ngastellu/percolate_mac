#!/usr/bin/env python

import numpy as np
import os



structypes = ['40x40', 'tempdot6', 'tempdot5']
motypes = ['lo', 'hi', 'virtual_w_HOMO']


for st in structypes:
    print(f'***** {st} *****')
    for mt in motypes:
        print(f'--- {mt} ---')
        datadir = f'/Users/nico/Desktop/simulation_outputs/percolation/{st}/MO_ipr_v_MO_cryst/{mt}'
        npys = os.listdir(datadir)
        for npy in npys:
            print(npy.split('.')[0], end = ' ')
            dat = np.load(os.path.join(datadir, npy))
            N = dat.shape[0]
            nMOs = N // 2
            iprs = dat[:nMOs]
            MO_crysts = dat[nMOs:]
            real_dat = np.vstack((iprs, MO_crysts))
            np.save(os.path.join(datadir, npy), real_dat)