#!/isr/bin/env python

from time import perf_counter
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from qcnico import plt_utils
from qcnico.qchemMAC import AO_gammas, MO_gammas
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from var_a_percolate_mac.MOs2sites import generate_sites_radii_list, LR_MOs, sites_mass
from var_a_percolate_mac.deploy_percolate import load_data


n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_type = sys.argv[3]

temps = np.arange(40,440,10)
dV = 0.0

pos, e, M, gamL, gamR = load_data(n, struc_type, mo_type, compute_gammas=True)
pos = pos[:,:2]
tree = KDTree(pos)

L, R = LR_MOs(gamL, gamR)

print('Generating sites and radii now...')
start = perf_counter()
centers, radii, ee, ii = generate_sites_radii_list(pos, M, L, R, e, hyperlocal='all')
end = perf_counter()
print(f'Done! [{end-start} seconds]')

masses = np.zeros(radii.shape[0])
r_sorted = np.zeros(radii.shape[0])
e_sorted = np.zeros(radii.shape[0])
k = 0
print('Total # of sites = ', radii.shape[0])
for n in range(M.shape[1]):
    jj = (ii == n)
    nsites = jj.sum()
    print(f'{n} ---> nsites = {nsites}')
    cc = centers[jj,:]
    rr = radii[jj]
    e_sorted[k:k+nsites] = ee[jj]
    psi = M[:,n]
    r_sorted[k:k+nsites] = rr
    masses[k:k+nsites] = sites_mass(psi,tree,cc,rr)
    k += nsites
    print(f'new k = {k}\n')

np.save('var_a_npys/ee.npy', ee)
np.save('var_a_npys/radii.npy', radii)
np.save('var_a_npys/centers.npy', centers)
np.save('var_a_npys/ii.npy', ii)
np.save('rr_v_masses_v_ee.npy',np.vstack((r_sorted,masses,e_sorted)))
