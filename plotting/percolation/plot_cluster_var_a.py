#!/usr/bin/env python

import sys
from os import path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico import plt_utils
from percolate import diff_arrs_var_a
from percplotting import plot_cluster_brute_force
from qcnico.coords_io import read_xyz
from qcnico.qchemMAC import MO_com, MO_rgyr

def read_pkl(run_ind,temp,datadir,pkl_prefix):
    #nsamples = len(run_inds)
    #ntemps = len(temps)
    sampdir = f"sample-{run_ind}"
    pkl = f"{pkl_prefix}-{temp}K.pkl"
    fo = open(path.join(datadir,sampdir,pkl),'rb')
    dat = pickle.load(fo)
    fo.close()
    return dat

def check_dists(A, centres, energies, dcrit, T, a0=30):
    print("Checking distances... dcrit = ", dcrit)
    kB = 8.617e-5
    ii, jj = A.nonzero()
    print(A.shape)
    for i, j in zip(ii,jj):
        dist = (np.abs(energies[i]) + np.abs(energies[j]) + np.abs(energies[i]-energies[j]))/(kB*T) + 2*np.linalg.norm(centres[i]-centres[j])/a0
        if dist > dcrit:
            print(f"!!! YIKES: u({i}, {j}) = {dist} !!!")




# ---------- MAIN ----------


structype = 'tempdot5'
nn = 10
motype = 'virtual_w_HOMO'
T = 300

runtype = 'sites'

if runtype not in ['MOs', 'sites']:
    print(f'Invalid run type {runtype}. Valid entries are:\n* "sites": for runs using sites created by k-clustering;\n*"MOs": for runs using the MOs directly as sites.\nExiting angrily.')
    sys.exit()


if structype == '40x40':
    rmax = 18.03
    synth_temp ='500'
elif structype == 'tempdot6':
    rmax=121.2
    synth_temp ='q400'
elif structype == 'tempdot5':
    rmax = 199.33
    synth_temp = '300'
else:
    print(f'Structure type {structype} is invalid. Exiting angrily.')
    sys.exit()

if runtype == 'sites':
    # sitesdir = f'/Users/nico/Desktop/simulation_outputs/percolation/{structype}/var_radii_data/to_local_sites_data_0.00105_psi_pow2_{motype}/sample-{nn}/'
    sitesdir = f'/Users/nico/Desktop/simulation_outputs/percolation/{structype}/var_radii_data/sites_data_rmax_{rmax}_{motype}/sample-{nn}/'

# pkl_prefix = f'out_percolate_rmax_{rmax}_psipow2_sites_gammas_{motype}'
pkl_prefix = f'out_percolate_rmax_{rmax}_{motype}'



datadir=path.expanduser(f"~/Desktop/simulation_outputs/percolation/{structype}")
rCC = 1.8


posdir = f'/Users/nico/Desktop/scripts/disorder_analysis_MAC/structures/sAMC-{synth_temp}/'
Mdir = path.join(datadir, f'MOs_ARPACK/{motype}/')
Sdir = f"/Users/nico/Desktop/simulation_outputs/percolation/site_ket_matrices/{structype}_rmax_{rmax}/"
edir = path.join(datadir, f'eARPACK/{motype}/')
# percdir = path.join(datadir, f'percolate_output/zero_field/to_local_rmax_{rmax}_psipow2_sites_gammas_{motype}/')
percdir = path.join(datadir, f'percolate_output/zero_field/rmax_{rmax}_{motype}/')


posfile = path.join(posdir, f'sAMC{synth_temp}-{nn}.xyz')

if motype == 'virtual':
    Mfile = path.join(Mdir,f'MOs_ARPACK_bigMAC-{nn}.npy')
    Sfile = path.join(Sdir, f'site_kets_psipow2-{nn}.npy')
    efile = path.join(edir, f'eARPACK_bigMAC-{nn}.npy')
else:   
    Sfile = path.join(Sdir, f'site_kets_{motype}-{nn}.npy')
    efile = path.join(edir, f'eARPACK_bigMAC-{nn}.npy')
    # efile = path.join(edir, f'eARPACK_{motype}_{structype}-{nn}.npy')

pos = read_xyz(posfile)

if runtype == 'MOs':
    M = np.load(Mfile)
    centres = MO_com(pos,M)
    radii = MO_rgyr(pos,M)
    energies = np.load(efile)
else: 
    M = np.load(Sfile)
    M /= np.linalg.norm(M,axis=0)
    centers = np.load(path.join(sitesdir, 'centers.npy'))
    radii = np.load(path.join(sitesdir, 'radii.npy'))
    energies = np.load(path.join(sitesdir, 'ee.npy'))



filter = radii < rmax
M = M[:,filter]
radii = radii[filter]
centers = centers[filter,:]
energies = energies[filter]

dat = read_pkl(nn,T,percdir,pkl_prefix)
clusters = dat[0]
print('Number of clusters = ', len(clusters))
c = np.array(list(clusters[0]))
dcrit = dat[1]
A = dat[2]
print('dcrit = ', dcrit)

plt_utils.setup_tex()
rcParams['font.size'] = 20

fig, ax = plt.subplots()
plot_cluster_brute_force(c,pos,M,A,show_densities=True, dotsize=0.9, usetex=True, show=False,rel_center_size=10.0,plt_objs=(fig,ax),centers=centers, inds = c)
# ax.set_title(f'MAC sample {nn}, $T = {T}$K')
plt.show()

