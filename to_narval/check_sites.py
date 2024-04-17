#!/usr/bin/env python


import numpy as np



with open('good_runs.txt') as fo:
    lines = fo.readlines()

good_inds = list(map(int, [l.strip() for l in lines]))



for i in good_inds:

    print(f'\n******** {i} ********')

    old_dir = f'old_bs/sample-{i}/'
    new_dir = f'sample-{i}/'

    ccnpy = 'cc.npy'
    eenpy = 'ee.npy'
    iinpy = 'ii.npy'

    old_cc = np.load(old_dir + ccnpy) 
    new_cc = np.load(new_dir + ccnpy) 
    print('Centers match = ', np.all(old_cc == new_cc))

    old_ee = np.load(old_dir + eenpy) 
    new_ee = np.load(new_dir + eenpy) 
    print('Energies match = ', np.all(old_ee == new_ee))

    old_ii = np.load(old_dir + iinpy)
    new_ii = np.load(new_dir + iinpy)
    print('Indices match = ', np.all(old_ii == new_ii))