#!/usr/bin/env python

import sys
import numpy as np
from time import perf_counter
from percolate_mac.deploy_percolate import load_data
from percolate_mac.MOs2sites import generate_site_list, generate_site_list_opt



n = int(sys.argv[1])
struc_type = 'tempdot6'
mo_type = 'virtual'
tolscal = 3.0

pos, e, M, gamL, gamR = load_data(n, struc_type, mo_type, compute_gammas=False)

gamL_tol = np.mean(gamL) + tolscal*np.std(gamL)
gamR_tol = np.mean(gamR) + tolscal*np.std(gamR)
L_mos = set((gamL > gamL_tol).nonzero()[0])
R_mos = set((gamR > gamR_tol).nonzero()[0])

print('Getting sites (old method)...')
start = perf_counter()
cc, ee, ii = generate_site_list(pos,M,L_mos,R_mos,e,nbins=100)
np.save('cc_old_method.npy', cc)
np.save('ee_old_method.npy', ee)
np.save('ii_old_method.npy', ii)
end = perf_counter()
print(f'Done! [{end-start} seconds]')

print('\nGetting sites (new method)...')
start = perf_counter()
cc, ee, ii = generate_site_list_opt(pos,M,L_mos,R_mos,e,nbins=100)
np.save('cc_new_method.npy', cc)
np.save('ee_new_method.npy', ee)
np.save('ii_new_method.npy', ii)
end = perf_counter()
print(f'Done! [{end-start} seconds]')
