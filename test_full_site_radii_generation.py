#!/isr/bin/env python

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from qcnico import plt_utils
from qcnico.qchemMAC import AO_gammas, MO_gammas
from qcnico.coords_io import read_xsf
from qcnico.remove_dangling_carbons import remove_dangling_carbons
from qcnico.qcplots import plot_MO
from MOs2sites import generate_sites_radii_list, LR_MOs, sites_mass



nsample = 64
gamma = 0.1

# percdir = f'/Users/nico/Desktop/simulation_outputs/percolation/Ata_structures/tempdot5/percolate_output/zero_field/virt_100x100_gridMOs/sample-{nsample}/'
Mdir = '/Users/nico/Desktop/simulation_outputs/percolation/40x40/MOs_ARPACK/virtual/'
posdir = '/Users/nico/Desktop/simulation_outputs/MAC_structures/pCNN/bigMAC_40x40_ensemble/'

M = np.load(Mdir + f'MOs_ARPACK_bigMAC-{nsample}.npy') 
energies = np.random.randn(M.shape[1])
# pos = np.load(posdir + f'coords-{nsample}.npy')
pos,_ = read_xsf(posdir + f'bigMAC-{nsample}_relaxed.xsf')
pos = remove_dangling_carbons(pos,1.8)

pos = pos[:,:2]
tree = KDTree(pos)

# agL, agR = AO_gammas(pos,gamma,brute_force=True)
# gamL, gamR = MO_gammas(M, agL, agR, return_diag=True) 

gamL = np.load('/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/zero_field/virt_100x100_gridMOs_var_a/sample-42/gamL_40x40-42_virtual.npy')
gamR = np.load('/Users/nico/Desktop/simulation_outputs/percolation/40x40/percolate_output/zero_field/virt_100x100_gridMOs_var_a/sample-42/gamR_40x40-42_virtual.npy')

L, R = LR_MOs(gamL, gamR)

eps_rho = 1.05e-3
flag_empty_clusters = True
max_r = 50.0

print('Generating sites and radii now...')
start = perf_counter()
centers, radii, ee, ii = generate_sites_radii_list(pos, M, L, R, energies, radii_rho_threshold=eps_rho)
np.save('ok_centers.npy', centers)
np.save('ok_radii.npy', radii)
np.save('ok_ii.npy', ii)
np.save('ok_ee.npy', ee)
# centers_hl, radii_hl, ee_hl, ii_hl = generate_sites_radii_list(pos, M, L, R, energies,hyperlocal='all', radii_rho_threshold=eps_rho,flag_empty_clusters=flag_empty_clusters,max_r=max_r)
# centers, radii, ee, ii = generate_sites_radii_list(pos, M, L, R, energies, radii_rho_threshold=eps_rho,flag_empty_clusters=flag_empty_clusters,max_r=max_r)
centers_hl, radii_hl, ee_hl, ii_hl = generate_sites_radii_list(pos, M, L, R, energies,hyperlocal='all', radii_rho_threshold=eps_rho,flag_empty_clusters=flag_empty_clusters,max_r=max_r)
end = perf_counter()
print(f'Done! [{end-start} seconds]')

print('Shape of centers: ', centers.shape)
print('Shape of ii: ', ii.shape)

masses = np.zeros(radii.shape[0])
r_sorted = np.zeros(radii.shape[0])
k = 0

masses_hl = np.zeros(radii.shape[0])
r_sorted_hl = np.zeros(radii.shape[0])
m = 0

for n in range(M.shape[1]):
    jj = (ii == n)
    nsites = jj.sum()
    cc = centers[jj,:]
    rr = radii[jj]
    psi = M[:,n]
    r_sorted[k:k+nsites] = rr
    masses[k:k+nsites] = sites_mass(psi,tree,cc,rr)
    k += nsites

    jj = (ii_hl == n)
    nsites = jj.sum()
    cc = centers_hl[jj,:]
    rr = radii_hl[jj]
    psi = M[:,n]
    r_sorted_hl[m:m+nsites] = rr
    masses_hl[m:m+nsites] = sites_mass(psi,tree,cc,rr)
    m += nsites
    
plt_utils.setup_tex()

plt_utils.multiple_histograms((radii,radii_hl),nbins=200,xlabel='Site radii [\AA]',labels=('$|\psi_n|^2$','$|\psi_n|^4$'))

fig, ax = plt.subplots()
ax.scatter(r_sorted,masses,c='r',s=4.0,alpha=0.6,label='$|\psi_n|^2$')
ax.scatter(r_sorted_hl,masses_hl,c='b',s=4.0,alpha=0.6,label='$|\psi_n|^4$')
ax.set_xlabel('Site radius [\AA]')
ax.set_ylabel('Site mass')
plt.legend()
plt.show()


imax_r = np.argmax(radii)
nn = ii[imax_r]

rel_centers =  centers[ii == nn]
rel_radii = radii[ii == nn]

print('REL RADII = ', rel_radii)

fig, ax = plt.subplots()
plot_MO(pos,M, nn, dotsize=1.0, plt_objs=(fig, ax), loc_centers=rel_centers, loc_radii=rel_radii)
fig, ax = plt.subplots()
plot_MO(pos, M, nn, dotsize=1.0, plt_objs=(fig, ax),show_rgyr=True)

# cc, *_ = get_MO_loc_centers_opt(pos, M, nn, 100, threshold_ratio=0.30)
# plot_MO(pos,M, nn, dotsize=1.0,loc_centers=cc)
# print('A priori centers = ', cc)
# psi = M[:,nn]
# clust_cc, ll, flagged_ll = assign_AOs(pos, cc, psi, psi_pow=4, flag_empty_clusters=True)
# print('Clusters centers = ', clust_cc)
# print(ll)
# print(flagged_ll)

# fs, fr = site_radii(pos,M,nn,ll,hyperlocal='sites',density_threshold=eps_rho,flagged_labels=flagged_ll,max_r=max_r)
# print('Final sites = ', fs)
# print('Final radii = ', fr)


# plot_MO(pos,M, nn, dotsize=1.0, loc_centers=fs, loc_radii=fr)

# masses = sites_mass(psi,tree,fs,fr)
# print('Masses = ', masses)

# cyc = rcParams['axes.prop_cycle'] #default plot colours are stored in this `cycler` type object
# clrs = [d['color'] for d in list(cyc[0:len(clust_cc)])]
# atom_clrs = [clrs[k] for k in ll]

# fig,ax = plt.subplots()
# fig, ax = plot_atoms(pos,dotsize=0.1,show=False,plt_objs=(fig,ax),usetex=False,colour=atom_clrs,zorder=1)
# ax.scatter(*clust_cc.T, c=clrs, marker='^', s=5.0)
# ax.scatter(*rel_centers.T, c='r',marker='h', s=5.0)
# plt.show()

np.random.seed(64)
for n in np.random.randint(M.shape[1], size=10):
    jj = (ii_hl == n)
    print(jj)
    nsites = jj.sum()
    cc = centers_hl[jj,:]
    print(cc)
    rr = radii_hl[jj]
    print(rr)
    plot_MO(pos,M, n, dotsize=1.0, loc_centers=cc, loc_radii=rr,c_rel_size=2)