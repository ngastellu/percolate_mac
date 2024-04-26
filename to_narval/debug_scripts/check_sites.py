#!/usr/bin/env python


import numpy as np



with open('good_runs.txt') as fo:
    lines = fo.readlines()

good_inds = list(map(int, [l.strip() for l in lines]))



with open('ii_mismatched_sites.txt', 'w') as fo:
    for i in good_inds:

        print(f'\n******** {i} ********')

        old_dir = f'old_bs/sample-{i}/'
        new_dir = f'sample-{i}/'

        ccnpy = 'cc.npy'
        eenpy = 'ee.npy'
        iinpy = 'ii.npy'

        old_cc = np.load(old_dir + ccnpy) 
        new_cc = np.load(new_dir + ccnpy)
        already_bad = False 
        try:
            print('Centers match = ', np.all(old_cc == new_cc))
        except Exception as e:
            print(e)
            fo.write(f'\n{i} (cc)')
            already_bad = True


        old_ee = np.load(old_dir + eenpy) 
        new_ee = np.load(new_dir + eenpy) 
        try:
            print('Energies match = ', np.all(old_cc == new_cc))
        except Exception as e:
            print(e)
            if already_bad:
                fo.write(' (ee)')
            else:
                fo.write(f'\n{i} (ee)')
                already_bad = True

        old_ii = np.load(old_dir + iinpy)
        new_ii = np.load(new_dir + iinpy)
        try:
            print('Centers match = ', np.all(old_cc == new_cc))
        except Exception as e:
            print(e)
            if already_bad:
                fo.write(' (ii)')
            else:
                fo.write(f'\n{i} (ii)')