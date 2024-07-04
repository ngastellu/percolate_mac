#!/usr/bin/env python

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.spatial import KDTree
from qcnico import plt_utils
from qcnico.qchemMAC import inverse_participation_ratios
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from qcnico.qcplots import plot_MO, plot_atoms
from MOs2sites import generate_sites_radii_list, LR_MOs, sites_mass, assign_AOs, get_MO_loc_centers_opt, site_radii, generate_sites_radii_list_naive, all_sites_ipr



nsample = 42
gamma = 0.1

# percdir = f'/Users/nico/Desktop/simulation_outputs/percolation/tempdot5/percolate_output/zero_field/virt_100x100_gridMOs/sample-{nsample}/'
Mdir = '/Users/nico/Desktop/simulation_outputs/percolation/tempdot5/MOs_ARPACK/virtual/'
posdir = '/Users/nico/Desktop/simulation_outputs/MAC_structures/Ata_structures/relaxed_structures/tempdot5/'

M = np.load(Mdir + f'MOs_ARPACK_bigMAC-{nsample}.npy') 
energies = np.random.randn(M.shape[1])
# pos = np.load(posdir + f'coords-{nsample}.npy')
# pos,_ = read_xsf(posdir + f'bigMAC-{nsample}_relaxed.xsf')
pos,_ = read_xsf(posdir + f'tempdot5n{nsample}_relaxed.xsf')
pos = remove_dangling_carbons(pos,1.8)
pos = pos[:,:2]
tree = KDTree(pos)

# agL, agR = AO_gammas(pos,gamma,brute_force=True)
# gamL, gamR = MO_gammas(M, agL, agR, return_diag=True) 

# gamL = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/zero_field/virt_100x100_gridMOs_var_a/sample-{nsample}/gamL_40x40-{nsample}_virtual.npy')
# gamR = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/zero_field/virt_100x100_gridMOs_var_a/sample-{nsample}/gamR_40x40-{nsample}_virtual.npy')

gamL = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/tempdot5/MO_gammas/gamL_40x40-{nsample}_virtual.npy')
gamR = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/tempdot5/MO_gammas/gamR_40x40-{nsample}_virtual.npy')

L, R = LR_MOs(gamL, gamR)

eps_rho = 1.05e-3
flag_empty_clusters = True
max_r = 50.0

# print('Generating sites and radii now...')
# start = perf_counter()
# centers, radii, ee, ii,labels = generate_sites_radii_list(pos, M, L, R, energies, radii_rho_threshold=eps_rho,flag_empty_clusters=flag_empty_clusters,max_r=max_r,return_labelled_atoms=True)
# # centers_hl, radii_hl, ee_hl, ii_hl, labels_hl = generate_sites_radii_list(pos, M, L, R, energies,hyperlocal='all', radii_rho_threshold=eps_rho,flag_empty_clusters=flag_empty_clusters,max_r=max_r,return_labelled_atoms=True)
# end = perf_counter()
# print(f'Done! [{end-start} seconds]')

# print('Shape of centers: ', centers.shape)
# print('Shape of ii: ', ii.shape)


# masses = np.zeros(radii.shape[0])
# r_sorted = np.zeros(radii.shape[0])
# k = 0

# # masses_hl = np.zeros(radii_hl.shape[0])
# # r_sorted_hl = np.zeros(radii_hl.shape[0])
# # m = 0

# for n in range(M.shape[1]):
#     psi = M[:,n]

#     jj = (ii == n)
#     nsites = jj.sum()
#     cc = centers[jj,:]
#     rr = radii[jj]
#     r_sorted[k:k+nsites] = rr
#     masses[k:k+nsites] = sites_mass(psi,tree,cc,rr)
#     k += nsites

#     # jj = (ii_hl == n)
#     # nsites = jj.sum()
#     # cc = centers_hl[jj,:]
#     # rr = radii_hl[jj]
#     # psi = M[:,n]
#     # r_sorted_hl[m:m+nsites] = rr
#     # masses_hl[m:m+nsites] = sites_mass(psi,tree,cc,rr) 
#     # m += nsites

# site_iprs = all_sites_ipr(M,labels,eps_rho=eps_rho)
# # site_iprs_hl = all_sites_ipr(M,labels_hl,eps_rho=eps_rho)

# print(site_iprs.shape)
# print(masses.shape)

# print(np.all(radii == r_sorted))

# plt_utils.setup_tex()

# # plt_utils.multiple_histograms((radii,radii_hl),nbins=200,xlabel='Site radii [\AA]',labels=('$|\psi_n|^2$','$|\psi_n|^4$'))

# fig, ax = plt.subplots()
# ax.scatter(r_sorted,masses,c='r',s=4.0,alpha=0.6,label='$|\psi_n|^2$')
# # ax.scatter(r_sorted_hl,masses_hl,c='b',s=4.0,alpha=0.6,label='$|\psi_n|^4$')
# # ax.scatter(r_sorted_hl,masses_hl,c='k',s=4.0,alpha=0.6,label='$|\psi_n|^4$')
# ax.set_xlabel('Site radius [\AA]')
# ax.set_ylabel('Site mass')
# plt.legend()
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(r_sorted,masses/(np.pi*r_sorted*r_sorted),c='r',s=4.0,alpha=0.6,label='$|\psi_n|^2$')
# # ax.scatter(r_sorted_hl,masses_hl,c='b',s=4.0,alpha=0.6,label='$|\psi_n|^4$')
# # ax.scatter(r_sorted_hl,masses_hl/(np.pi*r_sorted_hl*r_sorted_hl),c='k',s=4.0,alpha=0.6,label='$|\psi_n|^4$')
# ax.set_xlabel('Site radius [\AA]')
# ax.set_ylabel('Site density [\AA$^{-2}$]')
# plt.legend()
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(site_iprs,masses/(np.pi*r_sorted*r_sorted),c='r',s=4.0,alpha=0.6,label='$|\psi_n|^2$')
# # ax.scatter(r_sorted_hl,masses_hl,c='b',s=4.0,alpha=0.6,label='$|\psi_n|^4$')
# # ax.scatter(site_iprs_hl,masses_hl/(np.pi*r_sorted_hl*r_sorted_hl),c='k',s=4.0,alpha=0.6,label='$|\psi_n|^4$')
# ax.set_xlabel('Site IPR')
# ax.set_ylabel('Site density [\AA$^{-2}$]')
# plt.legend()
# plt.show()


# L_ipr = 1.0 / np.sqrt(site_iprs)
# densities = masses/(np.pi*r_sorted*r_sorted)

# high_L_ipr = np.argsort(L_ipr)[-5:]

# print('5 biggest sites (IPR-wise) = ' ,high_L_ipr)

# for imax_r in high_L_ipr:
#     big_ipr = L_ipr[imax_r]
#     bigr = radii[imax_r]
#     nn = ii[imax_r]

#     rel_centers =  centers[ii == nn]
#     rel_radii = radii[ii == nn]

#     fig, ax = plt.subplots()
#     c = ['r'] * rel_radii.shape[0]
#     ibig = (rel_radii == bigr).nonzero()[0][0]
#     c[ibig] = 'limegreen'
#     plot_MO(pos,M, nn, dotsize=1.0, plt_objs=(fig, ax), loc_centers=rel_centers, loc_radii=rel_radii, c_clrs = c)

# lo_density = np.argsort(densities)[:5]
# print('5 biggest sites (rho-wise) = ' ,lo_density
#       )

# for imax_r in lo_density:
#     lorho = densities[imax_r]
#     bigr = radii[imax_r]
#     nn = ii[imax_r]

#     rel_centers =  centers[ii == nn]
#     rel_radii = radii[ii == nn]

#     fig, ax = plt.subplots()
#     c = ['r'] * rel_radii.shape[0]
#     ibig = (rel_radii == bigr).nonzero()[0][0]
#     c[ibig] = 'limegreen'
#     plot_MO(pos,M, nn, dotsize=1.0, plt_objs=(fig, ax), loc_centers=rel_centers, loc_radii=rel_radii,c_clrs = c)

fig, ax = plt.subplots()
nn = 15
# rel_centers =  centers[ii == nn]
# rel_radii = radii[ii == nn]
plot_MO(pos, M, nn, dotsize=1.0, plt_objs=(fig, ax),show_rgyr=True, c_rel_size=300.0,scale_up=10.0,c_labels='$\langle \\bm{R} \\rangle$')

cc, *_ = get_MO_loc_centers_opt(pos, M, nn, 100, threshold_ratio=0.30)
plot_MO(pos,M, nn, dotsize=1.0,loc_centers=cc,c_rel_size=100.0,c_markers='^',c_clrs='r',scale_up=10.0,c_labels='a priori sites')
print('A priori centers = ', cc)
psi = M[:,nn]
clust_cc, ll, flagged_ll = assign_AOs(pos, cc, psi, psi_pow=4, flag_empty_clusters=True)
print('Clusters centers = ', clust_cc)
print(ll)
print(flagged_ll)

fs, fr = site_radii(pos,M,nn,ll,hyperlocal='sites',density_threshold=eps_rho,flagged_labels=flagged_ll,max_r=max_r)
print('Final sites = ', fs)
print('Final radii = ', fr)



masses = sites_mass(psi,tree,fs,fr)
print('Masses = ', masses)

cyc = rcParams['axes.prop_cycle'] #default plot colours are stored in this `cycler` type object
clrs = [d['color'] for d in list(cyc[0:len(clust_cc)])]
atom_clrs = [clrs[k] for k in ll]

fig,ax = plt.subplots()
fig, ax = plot_atoms(pos,dotsize=0.5,show=False,plt_objs=(fig,ax),usetex=False,colour=atom_clrs,zorder=1)
ax.scatter(*clust_cc[:-1].T, facecolors=clrs[:-1], marker='h', s=100.0, edgecolors='k',lw=0.5)
ax.scatter(*clust_cc[-1].T, facecolors=clrs[-1], marker='h', s=100.0, edgecolors='k',lw=0.5,label='k-cluster centres')
# ax.scatter(*rel_centers.T, c='r',marker='h', s=5.0)
ax.scatter(*cc[:-1].T, c='r',marker='^', s=100.0,edgecolors='k',lw=0.5)
ax.scatter(*cc[-1].T, c='r',marker='^', s=100.0,edgecolors='k',lw=0.5,label='a priori sites')
plt.legend()
plt.show()

plot_MO(pos,M, nn, dotsize=1.0, loc_centers=fs, loc_radii=fr,c_rel_size=100.0,scale_up=10.0,c_labels='new hopping sites')
# np.random.seed(64)
# for n in np.random.randint(M.shape[1], size=10):

#     fig, ax = plt.subplots()

#     jj = (ii == n)
#     print(jj)
#     nsites = jj.sum()
#     cc = centers[jj,:]
#     print(cc)
#     rr = radii[jj]
#     print(rr)
#     fig, ax = plot_MO(pos,M, n, dotsize=1.0, loc_centers=cc, loc_radii=rr,c_rel_size=2, plt_objs=(fig,ax), show=False)
    
#     jj = (ii_hl == n)
#     print(jj)
#     nsites = jj.sum()
#     cc = centers_hl[jj,:]
#     print(cc)
#     rr = radii_hl[jj]
#     print(rr)
#     fig, ax = plot_MO(pos,M, n, dotsize=1.0, loc_centers=cc, loc_radii=rr,c_rel_size=2,plt_objs=(fig,ax), show=False, c_clrs='limegreen')
#     plt.show()