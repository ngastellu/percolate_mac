#!/usr/bin/env python

from time import perf_counter
import os
import sys
import numpy as np
from scipy.spatial import KDTree
from qcnico.qchemMAC import AO_gammas, MO_gammas
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from var_a_percolate_mac.MOs2sites import generate_sites_radii_list, LR_MOs, sites_mass, sites_ipr
from var_a_percolate_mac.deploy_percolate import load_data


n = int(sys.argv[1])
struc_type = sys.argv[2]
mo_type = sys.argv[3]
psipow=2

pos, energies, M, gamL, gamR = load_data(n, struc_type, mo_type, gammas_method='compute', sort_energies=True)
N = pos.shape[0]

# Get rid of HOMO when generating sites; we only include it in the MOs and energies to readily compute eF
if mo_type == 'virtual_w_HOMO' and (N%2==0):
    energies = energies[1:]
    M = M[:,1:]
    gamL = gamL[1:]
    gamR = gamR[1:]

pos = pos[:,:2]
tree = KDTree(pos)

L, R = LR_MOs(gamL, gamR)


eps_rho = 1.05e-3
flag_empty_clusters = True
max_r = 50.0

print(f'_______________ eps_rho = {eps_rho} _______________')
print('Generating sites and radii now...')
start = perf_counter()
centers, radii, ee, ii,labels, site_matrix = generate_sites_radii_list(pos, M, L, R, energies, radii_rho_threshold=eps_rho,flag_empty_clusters=flag_empty_clusters,max_r=max_r,return_labelled_atoms=True,return_site_matrix=True, amplitude_pow=psipow)
end = perf_counter()
print(f'Done! [{end-start} seconds]')

npydir = f'sites_data_{mo_type}' 
if not os.path.isdir(npydir):
    os.mkdir(npydir)

np.save(os.path.join(npydir,'ee.npy'), ee)
np.save(os.path.join(npydir, 'radii.npy'), radii)
np.save(os.path.join(npydir, 'centers.npy'), centers)
np.save(os.path.join(npydir, 'ii.npy'), ii)
#    np.save(npydir + 'rr_v_masses_v_iprs_v_ee.npy',np.vstack((r_sorted,masses,site_iprs,e_sorted)))
np.save(os.path.join(npydir, 'site_state_matrix.npy'),site_matrix)