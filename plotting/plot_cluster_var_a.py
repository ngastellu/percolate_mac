#!/usr/bin/env python

from os import path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from qcnico import plt_utils
from percolate import diff_arrs_var_a, pair_inds
from qcnico.coords_io import read_xyz

def get_data(run_ind,temp,datadir,rmax):
    #nsamples = len(run_inds)
    #ntemps = len(temps)
    sampdir = f"sample-{run_ind}"
    pkl = f"out_percolate_rmax_{rmax}-{temp}K.pkl"
    fo = open(path.join(datadir,pkl),'rb')
    dat = pickle.load(fo)
    fo.close()
    return dat

# def get_center_pairs(darr, centres, d):
#     """Generator function that gets the hopping centers and the hopping center pairs to be plotted/connected
#     at each frame."""
#     N = centres.shape[0]
#     # while True:
#     connected_inds = (darr < d).nonzero()[0] #darr is 1D array     
#     ii, jj = pair_inds(connected_inds,N)
#     relevant_pairs = np.array([(centres[i], centres[j]) for i,j in zip(ii,jj)])
#     return relevant_pairs


# def update(frame):
#     """In this animation the frame is defined by which hopping site pairs are connected for a certain threshold 
#     distance d; those sites are then plotted, along with the edges connecting them."""
#     rel_pairs = frame
#     npairs = frame.shape[0]
#     rel_centers = np.unique(rel_pairs.reshape(2*npairs,2),axis=0)
#     ye = ax.scatter(pos.T[0], pos.T[1], c=rho, s=1.0, cmap='plasma',zorder=1)

#     ax.scatter(*rel_centers.T, marker='*', c='r', s=20.0, zorder=2)

#     for r1, r2 in rel_pairs:
#         pts = np.vstack((r1,r2)).T
#         ax.plot(*pts, 'r-', lw=1.0)
    
#     return ye


# Get data
nn = 64
structype = 'tempdot5'

if structype == '40x40':
    rmax = 18.03
    sT = '500'
elif structype == 'tempdot6':
    rmax = 121.2
    sT = 'q400'
else:
    rmax = 198.69
    sT = '300'

percdir=path.expanduser(f"~/Desktop/simulation_outputs/percolation/{structype}/percolate_output/zero_field/virt_100x100_gridMOs_rmax_{rmax}/sample-{nn}/")
sitesdir = f'/Users/nico/Desktop/simulation_outputs/percolation/{structype}/var_radii_data/to_local_sites_data/sample-{nn}/sites_data_0.00105/'



T = 200
kB = 8.617e-5
# posfile = path.join(f'/Users/nico/Desktop/simulation_outputs/MAC_structures/relaxed_no_dangle/{structype}/{structype}n{nn}_relaxed_no-dangle.xyz')
posfile = f"/Users/nico/Desktop/scripts/disorder_analysis_MAC/structures/sAMC-{sT}/sAMC{sT}-{nn}.xyz"

Mfile = path.join(f'/Users/nico/Desktop/simulation_outputs/percolation/{structype}/MOs_ARPACK/virtual/MOs_ARPACK_bigMAC-{nn}.npy')
efile = path.join(f'/Users/nico/Desktop/simulation_outputs/percolation/{structype}/eARPACK/virtual/eARPACK_bigMAC-{nn}.npy')
ccfile = path.join(sitesdir,'centers.npy')
rfile = path.join(sitesdir, f'radii.npy')
iifile = path.join(sitesdir,'ii.npy')
eefile = path.join(sitesdir,'ee.npy')


M = np.load(Mfile)
centres = np.load(ccfile)
MOinds = np.load(iifile)
energies = np.load(eefile)
radii = np.load(rfile)
pos = read_xyz(posfile)
dat = get_data(nn,T,percdir,rmax)
print(len(dat[0]))
c = dat[0][0] #sort cluster inds
rel_centers = centres[np.sort(list(c))]
# cluster_centres_inds = np.array([np.sum(i > c) for i in c])
# cluster_centres = centres[cluster_centres_inds]
dcrit = dat[1]
print(dcrit)

# Precompute necessary qties

edArr, rdArr,ij = diff_arrs_var_a(energies, centres, radii, eF=0)
darr = rdArr + (edArr/(kB*T))
print(darr)
#darr = dArray_logMA(energies, centres, T, a0=30,eF=0)
print(darr[darr<=dcrit].shape)
dmask = darr <= dcrit
rel_ds = darr[dmask]
rel_ij = ij[dmask]

rel_pairs = np.array([(centres[i], centres[j]) for i,j in rel_ij])
isites_unique = np.unique(rel_ij)
# rel_centers = centres[isites_unique]


rho = np.sum(M[:,np.unique(MOinds)]**2,axis=1)
scale_up = rho > 0.01
sizes = np.ones(rho.shape[0])
sizes[scale_up] *= 12


plt_utils.setup_tex()
rcParams['font.size'] = 40
# rcParams['figure.figsize'] = [19,9.5]

fig, ax = plt.subplots()

ye = ax.scatter(pos.T[0], pos.T[1], c=rho, s=sizes, cmap='plasma',zorder=1)


ax.scatter(*rel_centers.T, marker='*', c='r', s=20.0, zorder=2)

for r1r2, iijj in zip(rel_pairs, rel_ij):
    r1, r2 = r1r2
    i,j = iijj
    if i in c and j in c:
        pts = np.vstack((r1,r2)).T
        ax.plot(*pts, 'r-', lw=1.0)
    


# rcParams['figure.dpi'] = 200.0
# rcParams['figure.constrained_layout.use'] = True
ye = ax.scatter(pos.T[0], pos.T[1], c=rho, s=1.0, cmap='plasma',zorder=1)



cbar = fig.colorbar(ye,ax=ax,orientation='vertical')
ax.set_aspect('equal')
ax.set_xlabel("$x$ [\AA]")
ax.set_ylabel("$y$ [\AA]")
plt.show()

