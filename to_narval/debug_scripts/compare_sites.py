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

sane = np.all(iunique_new == iunique_old)

if sane:
    print('Unique inds match! ')

    ndiffs = np.zeros(iunique_old.shape[0], 'bool')
    nsame = np.ones(iunique_old.shape[0], 'bool')

    print('Looping over unique inds: ')
    for k, i in enumerate(iunique_old):
        print(f'*** {i} ***')
        match_new = (ii_new == i).nonzero()[0]
        match_old = (ii_old == i).nonzero()[0]

        nb_new = match_new.shape[0]
        nb_old = match_old.shape[0]

        print('Nb new: ', nb_new)
        print('Nb old: ', nb_old)

        if  nb_new == nb_old:
            sites_new = cc_new[match_new,:]
            sites_old = cc_old[match_old,:]
            same_sites = np.all(sites_new == sites_old)
            print(f'All sites match: ', same_sites)
            nsame[k] = same_sites

        else:
            ndiffs[k] = True

    bad_ii = iunique_old[ndiffs.nonzero()[0]] 
    bad_ii2 = iunique_old[~nsame]
    print('MOs with mismatched sites: ', bad_ii)
    print('MOs with same nb of sites but different site positions = ', bad_ii2)
    np.save('bad_ii.npy', bad_ii)
    np.save('bad_ii2.npy', bad_ii2)

else:
    print('Unique MO indices are different! Abort mission.') 