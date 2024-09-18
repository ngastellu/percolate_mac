#!/usr/bin/env python

import numpy as np
from qcnico.coords_io import read_xyz
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours
from qcnico.qcplots import plot_MO
import matplotlib.pyplot as plt
import pickle
from glob import glob


def get_struc_labels(ddir, prefix='sample',connector_char='-'):
    return np.array([int(x.split(connector_char)[-1]) for x in glob(f'{ddir}/{prefix}{connector_char}*')])

def zero_cryst_radii(nn, temps, ddir, rmax, dir_prefix='sample-', npy_prefix='clust_cryst-',return_isites=False):
    zero_cryst_radii = np.array([-1]) #initialise output arr for easier concatenation
    if return_isites: # !!!! ONLY use of working with a single structure !!!!
        all_izero_cryst = np.zeros(1,dtype='int')
    for n in nn:
        print(f'--------- {n} ---------')
        for T in temps:
            cluster_crysts = np.load(f'{ddir}/electronic_crystallinity/cluster_crystallinities_rmax_{rmax}/{dir_prefix}{n}/{npy_prefix}{T}K.npy')
            with open(f'{ddir}/percolate_output/zero_field/virt_100x100_gridMOs_rmax_{rmax}/sample-{n}/out_percolate_rmax_{rmax}-{T}K.pkl', 'rb') as fo:
                clusters = pickle.load(fo)[0]
            if len(clusters) != 1:
                print(len(clusters))
                continue # hard to deal with systems with multiple clusters; crystallinity unifies all clusters
            cluster = np.array(list(clusters[0]))
            radii = np.load(f'{ddir}/var_radii_data/to_local_sites_data/sample-{n}/sites_data_0.00105/radii.npy')
            izero_crysts_cluster = (cluster_crysts == 0) # indices of zero crystallinity sites in `cluster_crysts`
            # print(izero_crysts_cluster.sum())
            izero_crysts = cluster[izero_crysts_cluster] # actual indices of zero-crystallinity sites
            zero_cryst_radii = np.hstack((zero_cryst_radii, radii[izero_crysts]))
            if return_isites:
                all_izero_cryst = np.hstack((all_izero_cryst,izero_crysts))
    
    if return_isites:
        return zero_cryst_radii[1:], all_izero_cryst[1:]
    else:
        return zero_cryst_radii[1:] #get rid of spurious 1st element


 
structypes = ['40x40', 'tempdot6', 'tempdot5']
lbls= ['PixelCNN', '$\\tilde{T} = 0.6$', '$\\tilde{T} = 0.5$']
rmaxs = [18.03, 121.2, 198.69]
temps = np.arange(180,440,10)
clrs = MAC_ensemble_colours()

simdir = '/Users/nico/Desktop/simulation_outputs/'
percdir = simdir + 'percolation/'

setup_tex()
# fig, axs = plt.subplots(3,1,sharex=True)
nbins = 101
bins = np.linspace(0,100,nbins)

# for ax, st, r, lbl, c in zip(axs,structypes, rmaxs, lbls,clrs):

#     datadir = percdir + f'{st}' #electronic_crystallinity/cluster_crystallinities_rmax_{r}_renormd_by_ncryst_atoms/'
#     nn = get_struc_labels(datadir + f'/electronic_crystallinity/cluster_crystallinities_rmax_{r}')
#     print(nn)
#     # data = np.hstack([[np.load(f'sample-{n}/clust_cryst-{T}K.npy') for T in temps] for n in nn])
#     radii = zero_cryst_radii(nn, temps, datadir, r)
#     print(f'Max radius in {st} = {np.max(radii)}')
#     hist, bins = np.histogram(radii,bins=bins)
#     centers = (bins[1:] + bins[:-1]) * 0.5
#     dx = centers[1] - centers[0]

#     ax.bar(centers, hist,align='center',width=dx,color=c,label=lbl)
#     ax.legend()
#     ax.set_ylabel('Counts')
#     # ax.set_yscale('log')
#     # print(f'{st} ensemble has {ntiny} radii <= 1')

# ax.set_xlabel('Radii of zero-crystallinity conducting sites [\AA]')

# # plt.suptitle('Adjusted for number of crystalline atoms in structure')

# plt.show()


# np.random.seed(0)
structype = 'tempdot5'
nn = [13]
rmax = 198.69
datadir = percdir + '/' + structype
radii, isites = zero_cryst_radii(nn, temps, datadir, rmax, return_isites=True)
print(isites)
S = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/site_ket_matrices/{structype}_rmax_{rmax}/site_kets_psipow2-{nn[0]}.npy')
Sbinary = np.zeros(S.shape,dtype=int)
Sbinary[S.nonzero()] = 1
all_radii = np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/{structype}/var_radii_data/to_local_sites_data/sample-{nn[0]}/sites_data_0.00105/radii.npy')
centres =  np.load(f'/Users/nico/Desktop/simulation_outputs/percolation/{structype}/var_radii_data/to_local_sites_data/sample-{nn[0]}/sites_data_0.00105/centers.npy') 
pos = read_xyz(f'/Users/nico/Desktop/simulation_outputs/MAC_structures/relaxed_no_dangle/sAMC-300/sAMC300-{nn[0]}.xyz')

iii = np.argsort(radii)
ii = np.unique(isites[iii])
print(len(radii))
print(len(temps))
# print(np.all(radii[iii] == all_radii[ii]))

for i in ii:
    r = radii[i]
    c = centres[i]
    plot_MO(pos, Sbinary, i, dotsize=0.5,loc_centers=np.array([centres[i]]), loc_radii=[all_radii[i]],scale_up=20.0,cmap='bwr')