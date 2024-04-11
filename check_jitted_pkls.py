#!/usr/bin/env python

import pickle
import numpy as np


Ts = np.arange(100,150,10)

for T in Ts:
    print(f'\n\n*** Checking for T = {T}K ***')

    with open(f'out_jitted_percolate-{T}K.pkl', 'rb') as fo:
        pkl_jitted = pickle.load(fo)

    with open(f'out_percolate-{T}K.pkl', 'rb') as fo:
        pkl_vanilla = pickle.load(fo)
    
    print('Comparing cluster list:')
    clistj = pkl_jitted[0]
    clistv = pkl_vanilla[0]
    print('Nb. of clusters from jitted = ', len(clistj))
    print('Nb. of clusters from OG = ', len(clistv))
    if len(clistj) == len(clistv):
        k = 1
        for cj, cv in zip(clistj,clistv):
            print(f'Cluster {k} matches = ', cj==cv)
            k+=1
    else:
        print('!!! Nb of clusters differ !!!')
    
    dmatch = pkl_jitted[1] == pkl_vanilla[1]
    print('\nDistances match: ', dmatch)
    if not dmatch:
        print('Perc dist from jitted = ', pkl_jitted[1])
        print('Perc dist from OG = ', pkl_vanilla[1])
    
    print('\nFinal adjmats match = ', np.all(pkl_jitted[2] == pkl_vanilla[2]))
