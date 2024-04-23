#!/usr/bin/env python

import numpy as np

ee_old = np.load('ee_old_method.npy')
cc_old = np.load('cc_old_method.npy')
ii_old = np.load('ii_old_method.npy')

ee_new = np.load('ee_new_method.npy')
cc_new = np.load('cc_new_method.npy')
ii_new = np.load('ii_new_method.npy')


iunique_old = np.unique(ii_old)
iunique_new = np.unique(ii_new)

print('Unique inds match: ', np.all(iunique_new == iunique_old))

print('Looping over unique inds: ')
for i in iunique_old:
    print(f'*** {i} ***')
    match_new = (ii_new == i).nonzero()[0]
    match_old = (ii_old == i).nonzero()[0]

    e_new = ee_new[match_new]
    e_old = ee_old[match_old]

    print('Old energies consistent: ', np.all(e_old == e_old[0]))
    print('New energies consistent: ', np.all(e_new == e_new[0]))

    print('Old energy = New energy: ', e_old[0] == e_new[0])




    