#!/usr/bin/env python

import os
import sys
import numpy as np
from qcnico.qchemMAC import inverse_participation_ratios




ensemble = sys.argv[1]
eps_rho = 0.00105
psipow=2
motype = 'virtual'
renormalise = True



if ensemble == '40x40':
    rmax = 18.03
    lbls = np.arange(1,300)
elif ensemble == 'tempdot6':
    rmax = 121.2
    lbls = np.arange(132)
elif ensemble == 'tempdot5':
    rmax = 198.69
    lbls = np.arange(117)
else:
    print(f'{ensemble} is an invalid ensemble name.')


for n in lbls:

    print(f'**** Doing sample {n} ****')

    sites_datadir = f'sample-{n}/sites_data_{eps_rho}_psi_pow{psipow}/'
    
    try:
        M = np.load(os.path.expanduser(f'~/scratch/ArpackMAC/{ensemble}/MOs/{motype}/MOs_ARPACK_bigMAC-{n}.npy'))
        N = M.shape[0]
    except FileNotFoundError as e:
        print(e)
        continue

    try:
        S = np.load(sites_datadir + 'site_state_matrix.npy')
        radii = np.load(sites_datadir + 'radii.npy')
        ii = np.load(sites_datadir + 'ii.npy')
    except FileNotFoundError as e:
        print(e)
        continue
    
    mo_iprs = inverse_participation_ratios(M)
    site_iprs = inverse_participation_ratios(S)
    
    nsites = S.shape[1]
    out_all = np.zeros((nsites,2))

    rfilter = radii < rmax
    nsites_filtered = rfilter.sum()
    out_filtered = np.zeros((nsites_filtered,2))

    l = 0 #keeps track of sites w r < rmax
    for k in range(nsites):
        iMO = ii[k] # MO index
        out_all[k,0] = mo_iprs[iMO]
        out_all[k,1] = site_iprs[k]

        if rfilter[k]: # iff radii[k] < rmax
            out_filtered[l,0] = mo_iprs[iMO]
            out_filtered[l,1] = site_iprs[k]
            l += 1

    out_npy = f'sample-{n}/site_v_MO_iprs_all_sites.npy'
    out_npy_rmax_filtered = f'sample-{n}/site_v_MO_iprs_rmax_{rmax}.npy'

    np.save(out_npy, out_all)
    np.save(out_npy_rmax_filtered, out_filtered)