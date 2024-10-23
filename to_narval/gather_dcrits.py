#!/usr/bin/env python

from qcnico.data_utils import save_npy
import numpy as np
import os



structype = os.path.basename(os.getcwd())


if structype == '40x40':
    rmax = 18.03
    lbls = np.arange(1,301)
elif structype == 'tempdot6':
    rmax = 121.2
    lbls = np.arange(132)
elif structype == 'tempdot5':
    rmax = 198.69
    lbls = np.arange(117)
else:
    print(f'Invalid structure type {structype}. We outtie.')


T = 300
motypes = ['virtual', 'kBTlo', 'kBThi']

for mt in motypes:
    print(f'***** {mt} *****')

    dcrits = np.ones(lbls.shape[0]) * -1.0

    for k, n in enumerate(lbls):
        print(f'{n}', end=' ')
        try:
            dc = np.load(f'sample-{n}/rmax_{rmax}_sites_gammas_kBTlo_pkls/dcrits.npy')
            dcrits[k] = dc[1,dc[0] == T]
        except:
            print('<-- NPY is missing!', end='   ')


    succ = dcrits > 0 
    lbls_out = lbls[succ]
    dcrits = dcrits[succ]
    save_npy(np.vstack((lbls_out, dcrits)), f'dcrits_rmax_{rmax}_{T}K_{mt}.npy', 'dcrits')
