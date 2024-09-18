#!/usr/bin/env python

from var_a_percolate_mac.utils_analperc import get_conduction_clusters
import numpy as np
import sys


structype = sys.argv[1]
struct_mask_type = 'crystalline'
eps_rho = 0.00105
psipow = 2
cryst_renormalise = True

if structype == '40x40':
    rmax = 18.03
elif structype == 'tempdot6':
    rmax = 121.2
elif structype == 'tempdot5':
    rmax = 198.69
else:
    print(f'{structype} is an invalid structure type!')
    sys.exit()

run_lbl = f'rmax_{rmax}'
pkl_prefix = f'out_percolate_{run_lbl}'
temps = np.arange(40,440,10)



with open(f'to_local_{run_lbl}/good_runs_{run_lbl}.txt') as fo:
    lines = fo.readlines()

nn = [int(l.strip()) for l in lines]

for n in nn:
    print(f'****** Doing sample {n} ******',flush=True)
    # Load site kets, filter them according to radius criterion, and renormalise
    datadir = f'sample-{n}/'
    for T in temps:
        cl = get_conduction_clusters(datadir,pkl_prefix,T)
        if len(cl) == 0:
            print(T, end=' ',flush=True)
    print('\n')
