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
psipow = 2


pos, energies, M, gamL, gamR = load_data(n, struc_type, mo_type, compute_gammas=False)
pos = pos[:,:2]
tree = KDTree(pos)

L, R = LR_MOs(gamL, gamR)


eps_rho = 1.05e-3
flag_empty_clusters = True
max_r = 50.0
min_dist = 4.0
threshold_ratio_peaks = 0.05

print(f'_______________ eps_rho = {eps_rho} _______________')
print('Generating sites and radii now...')
start = perf_counter()
centers, radii, ee, ii,labels, site_matrix = generate_sites_radii_list(pos, M, L, R, energies, radii_rho_threshold=eps_rho,flag_empty_clusters=flag_empty_clusters,max_r=max_r,return_labelled_atoms=True,return_site_matrix=True, amplitude_pow=psipow,min_distance=min_dist,threshold_ratio=threshold_ratio_peaks)
end = perf_counter()
print(f'Done! [{end-start} seconds]')

#
#    print('Generating HYPERLOCAL sites and radii now...')
#    start = perf_counter()
#    centers_hl, radii_hl, ee_hl, ii_hl, labels_hl site_matrix_hl = generate_sites_radii_list(pos, M, L, R, energies,hyperlocal='all', radii_rho_threshold=eps_rho,flag_empty_clusters=flag_empty_clusters,max_r=max_r,return_labelled_atoms=True,return_site_matrix=True, amplitude_pow=1)
#    end = perf_counter()
#    print(f'Done! [{end-start} seconds]')
#
#    masses = np.zeros(radii.shape[0])
#    r_sorted = np.zeros(radii.shape[0])
#    e_sorted = np.zeros(radii.shape[0])
#    site_iprs = np.zeros(radii.shape[0])
#    k = 0
#
#    masses_hl = np.zeros(radii_hl.shape[0])
#    r_sorted_hl = np.zeros(radii_hl.shape[0])
#    e_sorted_hl = np.zeros(radii_hl.shape[0])
#    site_iprs_hl = np.zeros(radii_hl.shape[0])
#    m = 0
#    print('Total # of sites = ', radii.shape[0])
#    for n in range(M.shape[1]):
#        jj = (ii == n)
#        nsites = jj.sum()
#        print(f'{n} ---> nsites = {nsites}')
#        cc = centers[jj,:]
#        rr = radii[jj]
#        e_sorted[k:k+nsites] = ee[jj]
#        psi = M[:,n]
#        r_sorted[k:k+nsites] = rr
#        masses[k:k+nsites] = sites_mass(psi,tree,cc,rr)
#        site_iprs[k:k+nsites] = sites_ipr(psi,labels[n],eps_rho=eps_rho)
#        k += nsites
#        print(f'new k = {k}\n')
#
#        jj = (ii_hl == n)
#        nsites = jj.sum()
#        print(f'{n} ---> nsites = {nsites}')
#        cc = centers_hl[jj,:]
#        rr = radii_hl[jj]
#        e_sorted_hl[m:m+nsites] = ee_hl[jj]
#        psi = M[:,n]
#        r_sorted_hl[m:m+nsites] = rr
#        masses_hl[m:m+nsites] = sites_mass(psi,tree,cc,rr)
#        site_iprs_hl[m:m+nsites] = sites_ipr(psi,labels_hl[n],eps_rho=eps_rho)
#        m += nsites
    #print(f'new m = {m}\n')


npydir = f'sites_data_{eps_rho}_psi_pow{psipow}_many_peaks/' 
if not os.path.isdir(npydir):
    os.mkdir(npydir)

np.save(npydir + 'ee.npy', ee)
np.save(npydir + 'radii.npy', radii)
np.save(npydir + 'centers.npy', centers)
np.save(npydir + 'ii.npy', ii)
#    np.save(npydir + 'rr_v_masses_v_iprs_v_ee.npy',np.vstack((r_sorted,masses,site_iprs,e_sorted)))
np.save(npydir + 'site_state_matrix.npy',site_matrix)


#np.save(npydir + 'ee_hl.npy', ee_hl)
#np.save(npydir + 'radii_hl.npy', radii_hl)
#np.save(npydir + 'centers_hl.npy', centers_hl)
#np.save(npydir + 'ii_hl.npy', ii_hl)
#np.save(npydir + 'rr_v_masses_v_iprs_v_ee_hl.npy',np.vstack((r_sorted_hl,masses_hl,site_iprs_hl,e_sorted_hl)))
#np.save(npydir + 'site_state_matrix_hl.npy',site_matrix)
